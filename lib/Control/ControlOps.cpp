#include "Control/ControlDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace hail::control;

Block *DefContOp::body() { return &bodyRegion().back(); }

static LogicalResult verify(DefContOp op) {
  auto bodyTypes = op.body()->getArgumentTypes();
  auto contTypes = op.getType().cast<ContinuationType>().getInputTypes();

  if (contTypes != bodyTypes)
    return op.emitOpError(
      "expect the continuation body's arguments to have the same types "
      "as the continuation type's parameters");

  return success();
}

static LogicalResult verify(ApplyContOp op) { return success(); }

#define GET_OP_CLASSES
#include "Control/ControlOps.cpp.inc"
