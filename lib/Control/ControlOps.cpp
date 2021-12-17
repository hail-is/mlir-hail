#include "Control/ControlDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace hail::control;

Block *DefContOp::body() { return &bodyRegion().front(); }

static LogicalResult verify(DefContOp op) {
  auto bodyTypes = op.body()->getArgumentTypes();
  auto contArgTypes = op.getType().cast<ContinuationType>().getInputTypes();

  if (contArgTypes != bodyTypes)
    return op.emitOpError(
      "expects the continuation body's arguments to have the same types "
      "as the continuation type's parameters");

  return success();
}

static LogicalResult verify(ApplyContOp op) {
  auto contTypes = op.continuation().getType().cast<ContinuationType>().getInputTypes();
  auto argTypes = op.values().getTypes();

  if (contTypes != argTypes)
    return op.emitOpError("mismatched continuation and argument types");

  return success();
}

Block *CallCCOp::body() { return &bodyRegion().front(); }

static LogicalResult verify(CallCCOp op) {
  auto bodyTypes = op.body()->getArgumentTypes();
  if (bodyTypes.size() != 1) return op.emitOpError("body must take a single continuation");
  auto contType = bodyTypes[0].dyn_cast<ContinuationType>();
  if (!contType) return op.emitOpError("body must take a single continuation");
  auto contArgTypes = contType.getInputTypes();
  auto resultTypes = op.getResultTypes();

  if (resultTypes != contArgTypes)
    return op.emitOpError("mismatched continuation and result types");

  return success();
}

Block &IfOp::thenBlock() { return thenRegion().front(); }
Block &IfOp::elseBlock() { return elseRegion().front(); }

#define GET_OP_CLASSES
#include "Control/ControlOps.cpp.inc"
