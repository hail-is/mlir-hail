#ifndef HAIL_MLIR_DIALECT_OPTIONALTOSTANDARD_H
#define HAIL_MLIR_DIALECT_OPTIONALTOSTANDARD_H

#include <memory>
#include <vector>

namespace mlir {
struct LogicalResult;

class Pass;

class RewritePatternSet;
//using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace hail {
using namespace mlir;
void populateOptionalToStdConversionPatterns(RewritePatternSet &patterns);

/// Creates a pass to convert scf.for, scf.if and loop.terminator ops to CFG.
std::unique_ptr<Pass> createLowerOptionalToSTDPass();

} // namespace mlir

#endif //HAIL_MLIR_DIALECT_OPTIONALTOSTANDARD_H
