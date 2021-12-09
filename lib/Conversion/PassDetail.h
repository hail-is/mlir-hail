#ifndef HAIL_MLIR_DIALECT_PASSDETAIL_H
#define HAIL_MLIR_DIALECT_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/SCF.h"

namespace mlir {
class StandardOpsDialect;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // end namespace mlir
#endif //HAIL_MLIR_DIALECT_PASSDETAIL_H
