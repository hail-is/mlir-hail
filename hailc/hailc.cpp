#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "Conversion/Passes.h"

#include "Optional/OptionalDialect.h"
#include "Control/ControlDialect.h"

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum Action {
  None,
  DumpMLIR,
  DumpMLIRStd,
  DumpMLIRLLVM,
  DumpLLVMIR,
  EmitAsm,
  EmitObjectCode,
  // TODO DumpAsm,
  RunJIT,
};
}

static cl::opt<enum Action> action(
    "emit", cl::desc("Select the kind of output/behavior desired (Default: runs input with the JIT)"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRStd, "mlir-std",
                          "output the MLIR dump after optional to std lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(EmitAsm, "asm", "outbut the generated assembly")),
    cl::values(clEnumValN(EmitObjectCode, "obj", "output the compiled object file")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

cl::OptionCategory optFlags{"opt-like flags"};
cl::opt<bool> optO0{"O0",
                    cl::desc("Run opt passes and codegen at O0"),
                    cl::cat(optFlags)};
cl::opt<bool> optO1{"O1",
                    cl::desc("Run opt passes and codegen at O1"),
                    cl::cat(optFlags)};
cl::opt<bool> optO2{"O2",
                    cl::desc("Run opt passes and codegen at O2"),
                    cl::cat(optFlags)};
cl::opt<bool> optO3{"O3",
                    cl::desc("Run opt passes and codegen at O3"),
                    cl::cat(optFlags)};
cl::opt<bool> nativeCodegen{"native",
                            cl::desc("use native cpu features for code generation"),
                            cl::cat(optFlags)};

cl::OptionCategory clOptionsCategory{"linking options"};
cl::list<std::string> clSharedLibs{
    "shared-libs", cl::desc("Libraries to link dynamically"),
    cl::ZeroOrMore, cl::MiscFlags::CommaSeparated,
    cl::cat(clOptionsCategory)};

cl::opt<bool> dumpJitCode{"dump-jit-object-code",
                          cl::desc("when running jit code, also dump it")};

cl::opt<std::string> outputFilename{"o",
    cl::init("-"),
    cl::desc("Write any output, (mlir, llvm, object)")};
cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser> llvmPasses{
    cl::desc("LLVM optimizing passes to run"), cl::cat(optFlags)};

unsigned optLevel = 0, optPosition = 0;

static int loadAndProcess(std::unique_ptr<llvm::MemoryBuffer> inputFile,
                          mlir::MLIRContext &context,
                          mlir::OwningModuleRef &module) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  module = parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Could not parse input IR\n";
    return 1;
  }

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  bool isLowering = action >= DumpMLIRStd;
  if (optLevel > 0 || isLowering) {
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    // lightly optimize the base ir
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLowering) {
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    // partly lower optional to std/scf
    optPM.addPass(hail::createLowerOptionalToSTDPass());
  }

  bool isLoweringLLVM = action >= DumpMLIRLLVM;
  if (optLevel > 0 || isLoweringLLVM) {
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    // lightly optimize the partially lowered ir
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringLLVM) {
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();

    // lower SCF to cfg and lightly optimize
    optPM.addPass(mlir::createLowerToCFGPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    if (optLevel > 0) {
      optPM.addPass(mlir::createLoopFusionPass());
    }

    pm.addPass(mlir::createLowerToLLVMPass());
  }

  return mlir::failed(pm.run(*module)) ? 1 : 0;
}

static std::unique_ptr<llvm::TargetMachine> getTargetMachine() {
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    llvm::errs() << "NO target: " << errorMessage << "\n";
    return nullptr;
  }

  std::string cpu(nativeCodegen ? llvm::sys::getHostCPUName() : "generic");
  llvm::SubtargetFeatures features;
  if (nativeCodegen) {
    cpu = llvm::sys::getHostCPUName();
    llvm::StringMap<bool> hostFeatures;

    if (llvm::sys::getHostCPUFeatures(hostFeatures))
      for (auto &f : hostFeatures)
        features.AddFeature(f.first(), f.second);
  }

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine) {
    llvm::errs() << "Unable to create target machine\n";
  }
  return machine;
}

static std::unique_ptr<llvm::Module> translateToLLVMIRandOptimize(llvm::LLVMContext &llvmContext,
                                                                  ArrayRef<const llvm::PassInfo *> passes,
                                                                  llvm::TargetMachine *targetMachine,
                                                                  mlir::ModuleOp module)
{
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetMachine->getTargetTriple().str());

  auto optPipeline = mlir::makeLLVMPassesTransformer(
      passes, optLevel, targetMachine, optPosition);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }

  return llvmModule;
}

static int compileStatic(llvm::raw_fd_ostream &os,
                         ArrayRef<const llvm::PassInfo *> passes,
                         mlir::ModuleOp module) {
  // fresh llvm context
  llvm::LLVMContext llvmContext;

  auto target = getTargetMachine();
  if (!target) return 1;

  auto llvmModule = translateToLLVMIRandOptimize(llvmContext, passes, target.get(), module);
  if (!llvmModule) return 1;

  auto cgft = llvm::CGFT_Null;

  switch (action) {
    case Action::DumpLLVMIR: os << *llvmModule << '\n'; os.flush(); return 0;
    case Action::EmitAsm: cgft = llvm::CGFT_AssemblyFile; break;
    case Action::EmitObjectCode: cgft = llvm::CGFT_ObjectFile; break;
    default: assert(false && "invalid action type");
  }

  llvm::legacy::PassManager pm;
  if (target->addPassesToEmitFile(pm, os, nullptr, cgft)) {
    llvm::errs() << "Can't emit " << (cgft == llvm::CGFT_AssemblyFile ? "assembly" : "object")
        << " file for the target machine\n";
    return 1;
  }

  pm.run(*llvmModule);
  os.flush();
  return 0;
}

static int runJIT(ArrayRef<const llvm::PassInfo *> passes, mlir::ModuleOp module) {
  return 1;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::initializeLLVMPasses();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "hail compiler\n");

  if (action == Action::None) {
    auto fname = llvm::StringRef(outputFilename);
    if (fname == "-" /* default */) {
      action = Action::RunJIT;
    } else if (fname.endswith(".mlir")) {
      action = Action::DumpMLIR;
    } else if (fname.endswith(".ll")) {
      action = Action::DumpLLVMIR;
    } else if (fname.endswith(".s") || fname.endswith(".S")) {
      action = Action::EmitAsm;
    } else /* if (fname.endswith(".o")) */ {
      action = Action::EmitObjectCode;
    } // else { FIXME link an executable }
  }

  SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      optO0, optO1, optO2, optO3};

  unsigned optCLIPosition = 0;
  // Determine if there is an optimization flag present, and its CLI position
  // (optCLIPosition).
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optLevel = j;
      optCLIPosition = flag.getPosition();
      break;
    }
  }

  // Generate vector of pass information, plus the index at which we should
  // insert any optimization passes in that vector (optPosition).
  SmallVector<const llvm::PassInfo *, 4> passes;
  for (unsigned i = 0, e = llvmPasses.size(); i < e; ++i) {
    passes.push_back(llvmPasses[i]);
    if (optCLIPosition < llvmPasses.getPosition(i)) {
      optPosition = i;
      optCLIPosition = UINT_MAX; // To ensure we never insert again
    }
  }

  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << '\n';
    return 1;
  }

  if (outputFilename == "-"
      && (action == Action::EmitObjectCode
          || (action == Action::RunJIT && dumpJitCode)))
  {
    outputFilename = inputFilename + ".o";
  }

  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<hail::optional::OptionalDialect,
                  hail::control::ControlDialect,
                  mlir::StandardOpsDialect,
                  mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);

  MLIRContext context(registry);

  mlir::OwningModuleRef module;
  if (int error = loadAndProcess(std::move(inputFile), context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = action <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->print(outputFile->os());
    return 0;
  }

  if (action != Action::RunJIT) {
    int ret = compileStatic(outputFile->os(), passes, *module);
    if (!ret) outputFile->keep();
    return ret;
  }

  return runJIT(passes, *module);
}
