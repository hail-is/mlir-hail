#ifndef HAIL_CONVERSION_PASSES
#define HAIL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// OptionalToStandard
//===----------------------------------------------------------------------===//

def OptionalToStandard : Pass<"convert-optional-to-std"> {
  let summary = "Convert Optional dialect to Standard dialect";
  let constructor = "hail::createLowerOptionalToSTDPass()";
  let dependentDialects = ["StandardOpsDialect", "scf::SCFDialect", "LLVM::LLVMDialect"];
}

#endif // HAIL_CONVERSION_PASSES
