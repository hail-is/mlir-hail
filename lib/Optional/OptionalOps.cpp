#include "Optional/OptionalDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

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

struct RemoveConstructConsumeOpt : public OpRewritePattern<ConsumeCoOptOp> {
  using OpRewritePattern<ConsumeCoOptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConsumeCoOptOp op,
                                PatternRewriter &rewriter) const override {
      if (auto construct = op.opt().getDefiningOp<ConstructCoOptOp>()) {
        if (!op.opt().hasOneUse()) {
            return failure();
        }
        Block *block = &construct.reg().front();
        rewriter.mergeBlockBefore(block, op, op.getOperands().slice(1, 2));
        rewriter.eraseOp(op);
        rewriter.eraseOp(construct);
        return success();
      }

      return failure();
  };
};

struct RemoveUnpackPack : public OpRewritePattern<UnpackOptionalOp> {
  using OpRewritePattern<UnpackOptionalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnpackOptionalOp unpack,
                                PatternRewriter &rewriter) const override {

    if (auto pack = unpack.input().getDefiningOp<PackOptionalOp>()) {
      rewriter.replaceOp(unpack, pack.getOperands());
      return success();
    }

    return failure();
  }
};

struct RemoveCoOptToOptConversion : public OpRewritePattern<CoOptToOptOp> {
  using OpRewritePattern<CoOptToOptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoOptToOptOp op,
                                PatternRewriter &rewriter) const override {
    if (auto optToCoOpt = op.input().getDefiningOp<OptToCoOptOp>()) {
      rewriter.replaceOp(op, optToCoOpt.input());
      return success();
    }

    return failure();
  }
};

} // namespace

void ConsumeOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
    results.add<RemoveConsumePresentOrMissing>(context);
}

void UnpackOptionalOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<RemoveUnpackPack>(context);
}

void ConsumeCoOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
    results.add<RemoveConstructConsumeOpt>(context);
}

void CoOptToOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
    results.add<RemoveCoOptToOptConversion>(context);
}

static LogicalResult verify(ConsumeOptOp op) {
    auto inputTypes = op.input().getType().cast<OptionalType>().getValueTypes();
    auto presentTypes = op.presentBlock()->getArgumentTypes();

    if (inputTypes != presentTypes)
        return op.emitOpError(
            "expect the presentBlock's arguments to have the same types "
            "as the consumed option value");

    return RegionBranchOpInterface::verifyTypes(op);
}

Block *ConsumeOptOp::missingBlock() { return &missingRegion().back(); }
YieldOp ConsumeOptOp::missingYield() { return cast<YieldOp>(&missingBlock()->back()); }
Block *ConsumeOptOp::presentBlock() { return &presentRegion().back(); }
YieldOp ConsumeOptOp::presentYield() { return cast<YieldOp>(&presentBlock()->back()); }

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ConsumeOptOp::getSuccessorRegions(Optional<unsigned> index,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `missing` and the `present` regions branch back to the parent operation.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  regions.push_back(RegionSuccessor(&missingRegion()));
  regions.push_back(RegionSuccessor(&presentRegion()));
}

// TableGen'erated class code

#define GET_OP_CLASSES
#include "Optional/OptionalOps.cpp.inc"
