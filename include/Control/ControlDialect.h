#ifndef HAIL_CONTROLDIALECT_H
#define HAIL_CONTROLDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "Control/ControlOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Control/ControlOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Control/ControlOps.h.inc"

#endif
