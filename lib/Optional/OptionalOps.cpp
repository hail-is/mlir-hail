#include "Control/ControlDialect.h"
#include "Optional/OptionalDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace hail;
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
        Block &block = construct.body();
        rewriter.mergeBlockBefore(&block, op, op.getOperands().slice(1, 2));
        rewriter.eraseOp(op);
        rewriter.eraseOp(construct);
        return success();
      }

      return failure();
  };
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

struct RemoveOptToCoOptConversion : public OpRewritePattern<OptToCoOptOp> {
  using OpRewritePattern<OptToCoOptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OptToCoOptOp op,
                                PatternRewriter &rewriter) const override {
    auto opt = op.input();
    if (auto coOptToOpt = opt.getDefiningOp<CoOptToOptOp>()) {
      if (opt.hasOneUse()) {
        rewriter.replaceOp(op, coOptToOpt.input());
        rewriter.eraseOp(coOptToOpt);
        return success();
      }
    }

    return failure();
  }
};

} // namespace

void ConsumeOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
    results.add<RemoveConsumePresentOrMissing>(context);
}

void ConsumeCoOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
    results.add<RemoveConstructConsumeOpt>(context);
}

void CoOptToOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
    results.add<RemoveCoOptToOptConversion>(context);
}

void OptToCoOptOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<RemoveOptToCoOptConversion>(context);
}

static LogicalResult verify(ConsumeOptOp op) {
    auto inputTypes = op.input().getType().cast<OptionalType>().getValueTypes();
    auto presentTypes = op.presentBlock().getArgumentTypes();

    if (inputTypes != presentTypes)
        return op.emitOpError(
            "expect the presentBlock's arguments to have the same types "
            "as the consumed option value");

    return RegionBranchOpInterface::verifyTypes(op);
}

static LogicalResult verify(ConstructCoOptOp op) {
  auto coOptType = op.getType();
  auto valueTypes = coOptType.getValueTypes();
  if (op.body().getNumArguments() != 2)
    return op.emitOpError("expects the body to have exactly two arguments");
  auto missingContType = op.body().getArgument(0).getType().dyn_cast<control::ContinuationType>();
  if (!missingContType || !missingContType.getInputTypes().empty())
    return op.emitOpError("expects the first body argument to be a missing continuation with zero args");
  auto presentContType = op.body().getArgument(1).getType().dyn_cast<control::ContinuationType>();
  if (!presentContType || presentContType.getInputTypes() != valueTypes)
    return op.emitOpError("expects the second body argument to be a present continuation with arg types "
                          "matching the value types of the result CoOptional type");
  return success();
}

static LogicalResult verify(ConsumeCoOptOp op) {
  auto coOptType = op.opt().getType().cast<CoOptionalType>();
  auto valueTypes = coOptType.getValueTypes();
  auto missingContType = op.missing().getType().dyn_cast<control::ContinuationType>();
  if (!missingContType || !missingContType.getInputTypes().empty())
    return op.emitOpError("expects the second argument to be a missing continuation with zero args");
  auto presentContType = op.present().getType().dyn_cast<control::ContinuationType>();
  if (!presentContType || presentContType.getInputTypes() != valueTypes)
    return op.emitOpError("expects the third argument to be a present continuation with arg types "
                          "matching the value types of the first arg's CoOptional type");
  return success();
}

Block &ConsumeOptOp::missingBlock() { return missingRegion().front(); }
YieldOp ConsumeOptOp::missingYield() { return cast<YieldOp>(missingBlock().back()); }
Block &ConsumeOptOp::presentBlock() { return presentRegion().front(); }
YieldOp ConsumeOptOp::presentYield() { return cast<YieldOp>(presentBlock().back()); }

Block &ConstructCoOptOp::body() { return bodyRegion().back(); }

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
