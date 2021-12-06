#ifndef HAIL_OPTIONAL_OPTIONALOPS_H
#define HAIL_OPTIONAL_OPTIONALOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Optional/OptionalOps.h.inc"

#endif
