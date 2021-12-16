#ifndef HAIL_MLIR_DIALECT_OPTIONALTOSTANDARD_H
#define HAIL_MLIR_DIALECT_OPTIONALTOSTANDARD_H

#include <memory>

namespace mlir {
struct LogicalResult;

class Pass;

class RewritePatternSet;
} // namespace mlir

namespace hail {
using namespace mlir;
void populateOptionalToStdConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createLowerOptionalToSTDPass();

} // namespace mlir

#endif //HAIL_MLIR_DIALECT_OPTIONALTOSTANDARD_H
