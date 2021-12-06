#include "Optional/OptionalOps.h"
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
struct RemoveConsumeMissing : public OpRewritePattern<ConsumeOptOp> {
    using OpRewritePattern<ConsumeOptOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConsumeOptOp op,
                                  PatternRewriter &rewriter) const override {
        auto missing = op.input().getDefiningOp<MissingOp>();
        if (!missing)
            return failure();

        replaceOpWithRegion(rewriter, op, op.missingRegion());

        return success();
    }
};

} // namespace

void ConsumeOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
    results
        .add<RemoveConsumeMissing>(context);
}
