#include "Control/ControlDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "Control/ControlOps.cpp.inc"

using namespace mlir;
using namespace hail::control;
