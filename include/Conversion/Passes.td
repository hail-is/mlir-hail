#ifndef HAIL_CONVERSION_PASSES
#define HAIL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"
include "Control/ControlDialect.td"

//===----------------------------------------------------------------------===//
// ControlToStandard
//===----------------------------------------------------------------------===//

def ControlToStandard : Pass<"convert-control-to-std"> {
  let summary = "Convert Control dialect to Standard dialect";
  let constructor = "hail::createLowerControlToSTDPass()";
  let dependentDialects = ["mlir::StandardOpsDialect"];
}

//===----------------------------------------------------------------------===//
// OptionalToStandard
//===----------------------------------------------------------------------===//

def OptionalToStandard : Pass<"convert-optional-to-std"> {
  let summary = "Convert Optional dialect to Standard dialect";
  let constructor = "hail::createLowerOptionalToSTDPass()";
  let dependentDialects = ["mlir::StandardOpsDialect", "control::ControlDialect", "mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
}

#endif // HAIL_CONVERSION_PASSES
