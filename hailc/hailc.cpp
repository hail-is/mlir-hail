#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "Conversion/Passes.h"

#include "Optional/OptionalDialect.h"
#include "Control/ControlDialect.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum Action {
  None,
  DumpMLIR,
  DumpMLIRLLVM,
  DumpLLVMIR,
  DumpObjectCode,
  DumpAsm,
  RunJIT
};
}

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output/behavior desired (Default: runs input with the JIT)"),
    cl::init(RunJIT),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(DumpAsm, "asm", "outbut the generated assembly")),
    cl::values(clEnumValN(DumpObjectCode, "object", "output the compiled object file")),
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
cl::OptionCategory clOptionsCategory{"linking options"};
cl::list<std::string> clSharedLibs{
    "shared-libs", cl::desc("Libraries to link dynamically"),
    cl::ZeroOrMore, cl::MiscFlags::CommaSeparated,
    cl::cat(clOptionsCategory)};

cl::opt<std::string> outputFilename{"o",
    cl::desc("Write any output, (mlir, llvm, asm, object)")};
cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser> llvmPasses{
    cl::desc("LLVM optimizing passes to run"), cl::cat(optFlags)};

int loadAndProcess(mlir::MLIRContext &context,
                   mlir::OwningModuleRef &module)
{
  // TODO
  return -1;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/0 /*TODO respect command line opt level*/, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  // TODO alternate output stream
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::initializeLLVMPasses();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "hail compiler\n");

  mlir::DialectRegistry registry;
  registry.insert<hail::optional::OptionalDialect>();
  registry.insert<hail::control::ControlDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningModuleRef module;
  if (int error = loadAndProcess(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    // TODO output file
    module->dump();
    return 0;
  }

  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  // TODO object emission, execution
}
