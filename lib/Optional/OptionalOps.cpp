#include "Optional/OptionalDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "Optional/OptionalOps.cpp.inc"

using namespace mlir;
using namespace hail::optional;

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
/// Cribbed from SCF.cpp
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
    assert(llvm::hasSingleElement(region) && "expected single-block region");
    Block *block = &region.front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.mergeBlockBefore(block, op, blockArgs);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
}

namespace {
struct RemoveConsumePresentOrMissing : public OpRewritePattern<ConsumeOptOp> {
    using OpRewritePattern<ConsumeOptOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConsumeOptOp op,
                                  PatternRewriter &rewriter) const override {

        if (op.input().getDefiningOp<MissingOp>()) {
            replaceOpWithRegion(rewriter, op, op.missingRegion());
            return success();
        }

        if (auto present = op.input().getDefiningOp<PresentOp>()) {
            replaceOpWithRegion(rewriter, op, op.presentRegion(), present.getOperands());
            return success();
        }

        return failure();
    }
};

} // namespace

void ConsumeOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
    results
        .add<RemoveConsumePresentOrMissing>(context);
}

LogicalResult PresentOp::inferReturnTypes(MLIRContext *context,
                                          llvm::Optional<Location> location,
                                          ValueRange operands,
                                          DictionaryAttr attributes,
                                          RegionRange regions,
                                          llvm::SmallVectorImpl<Type>&inferredReturnTypes) {
    Type resultType = OptionalType::get(context, operands[0].getType());
    inferredReturnTypes.push_back(resultType);
}