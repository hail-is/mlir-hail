#ifndef HAIL_OPTIONALDIALECT_H
#define HAIL_OPTIONALDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Optional/OptionalOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Optional/OptionalOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Optional/OptionalOps.h.inc"

#endif
