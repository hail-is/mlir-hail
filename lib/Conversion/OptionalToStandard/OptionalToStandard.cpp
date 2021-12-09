#include "Conversion/OptionalToStandard/OptionalToStandard.h"
#include "../PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Optional/OptionalDialect.h"


using namespace mlir;
using namespace hail;
using namespace hail::optional;

namespace {

struct OptionalToStandardPass : public OptionalToStandardBase<OptionalToStandardPass> {
  void runOnOperation() override;
};

struct ConvertPresentOp : public OpConversionPattern<PresentOp> {
  using OpConversionPattern<PresentOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PresentOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
    auto constTrue = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), true));
    rewriter.replaceOpWithNewOp<PackOptionalOp>(op, op.getType(), constTrue.getResult(), operands[0]);

    return success();
  }
};

struct ConvertMissingOp : public OpConversionPattern<MissingOp> {
  using OpConversionPattern<MissingOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MissingOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
    auto constFalse = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), false));
    auto undefined = rewriter.create<UndefinedOp>(op.getLoc(), op.getType().getValueType());
    rewriter.replaceOpWithNewOp<PackOptionalOp>(op, op.getType(), constFalse.getResult(), undefined);

    return success();
  }
};

struct ConvertConsumeOptOp : public OpRewritePattern<ConsumeOptOp> {
  using OpRewritePattern<ConsumeOptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConsumeOptOp op,
                                PatternRewriter &rewriter) const {
    auto unpack = rewriter.create<UnpackOptionalOp>(op.getLoc(), rewriter.getI1Type(), op.input().getType().cast<OptionalType>().getValueType(), op.input());
    auto ifOp = rewriter.replaceOpWithNewOp<scf::IfOp>(op, op.getResultTypes(), unpack.isDefined(), /* withElseRegion= */ true);
    rewriter.mergeBlocks(&op.missingRegion().front(), ifOp.elseBlock());
    rewriter.mergeBlocks(&op.presentRegion().front(), ifOp.thenBlock(), unpack.value());
    // replace optional.yield with scf.yield
    auto elseYield = ifOp.elseBlock()->getTerminator();
    rewriter.setInsertionPointToEnd(ifOp.elseBlock());
    rewriter.replaceOpWithNewOp<scf::YieldOp>(elseYield, elseYield->getOperands());
    auto thenYield = ifOp.thenBlock()->getTerminator();
    rewriter.setInsertionPointToEnd(ifOp.thenBlock());
    rewriter.replaceOpWithNewOp<scf::YieldOp>(thenYield, thenYield->getOperands());

    return success();
  }
};

}

void hail::populateOptionalToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertPresentOp, ConvertMissingOp, ConvertConsumeOptOp>(patterns.getContext());
}

void OptionalToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateOptionalToStdConversionPatterns(patterns);
  // Configure conversion to lower out .... Anything else is fine.
  ConversionTarget target(getContext());
  target.addIllegalOp<PresentOp, MissingOp, ConsumeOptOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> hail::createLowerOptionalToSTDPass() {
  return std::make_unique<OptionalToStandardPass>();
}
