#include "Conversion/OptionalToStandard/OptionalToStandard.h"
#include "../PassDetail.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

struct ConvertUndefinedOp : public OpRewritePattern<UndefinedOp> {
  using OpRewritePattern<UndefinedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UndefinedOp op,
                                PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, op.result().getType());
    return success();
  }
};

struct ConvertIfReturningOptional : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  void transferBody(Block *source, Block *dest, ArrayRef<OpResult> optResults, PatternRewriter &rewriter) const {
    Type i1Type = rewriter.getI1Type();
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest);
    // Insert unpack operations and update the yielded results
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    rewriter.startRootUpdate(yieldOp);
    for (auto en : llvm::enumerate(optResults)) {
      auto i = en.index();
      auto result = en.value();
      unsigned int newIdx = result.getResultNumber() + i;
      auto unpack = rewriter.create<UnpackOptionalOp>(yieldOp->getLoc(), i1Type,
                                        result.getType().cast<OptionalType>().getValueType(),
                                        yieldOp->getOperand(newIdx));
      yieldOp->setOperands(newIdx, 1, unpack.getResults());
    }
    rewriter.finalizeRootUpdate(yieldOp);
  }

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const {
    Type i1Type = rewriter.getI1Type();
    // compute the set of optional results, and the new expanded result types
    SmallVector<OpResult, 4> optResults;
    SmallVector<Type, 4> newResultTypes;
    for (auto result : ifOp.getResults()) {
      Type resultType = result.getType();
      if (auto optResultType = resultType.dyn_cast<OptionalType>()) {
        Type valueType = optResultType.getValueType();
        optResults.push_back(result);
        newResultTypes.push_back(i1Type);
        newResultTypes.push_back(valueType);
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    // Create a replacement operation with empty then and else regions.
    auto emptyBuilder = [](OpBuilder &, Location) {};
    auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes, ifOp.condition(),
                                            /* withElseRegion = */ true);

    // Move the bodies and update the terminators
    transferBody(ifOp.getBody(0), newOp.getBody(0), optResults, rewriter);
    transferBody(ifOp.getBody(1), newOp.getBody(1), optResults, rewriter);
    rewriter.setInsertionPoint(ifOp);

    // Add pack operations after the if, and compute the values to replace the old if
    SmallVector<Value, 4> repResults;
    for (auto en : llvm::enumerate(optResults)) {
      auto i = en.index();
      auto result = en.value();
      unsigned int newIdx = result.getResultNumber() + i;
      Type resultType = result.getType();
      if (auto optResultType = resultType.dyn_cast<OptionalType>()) {
        Type valueType = optResultType.getValueType();
        auto pack = rewriter.create<PackOptionalOp>(ifOp->getLoc(), optResultType,
                                                   newOp.getResult(newIdx), newOp.getResult(newIdx + 1));
        repResults.push_back(pack.result());
      } else {
        repResults.push_back(result);
      }
    }

    // Replace the operation by the new one.
    rewriter.replaceOp(ifOp, repResults);

    return success();
  }
};

}

void hail::populateOptionalToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertPresentOp, ConvertMissingOp, ConvertConsumeOptOp,
               ConvertIfReturningOptional>(patterns.getContext());
}

void OptionalToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateOptionalToStdConversionPatterns(patterns);
  // Configure conversion to lower out .... Anything else is fine.
  ConversionTarget target(getContext());
  target.addIllegalOp<PresentOp, MissingOp, ConsumeOptOp>();
  target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp op) {
    for (auto result : op.results()) {
      if (result.getType().isa<OptionalType>()) return false;
    }
    return true;
  });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> hail::createLowerOptionalToSTDPass() {
  return std::make_unique<OptionalToStandardPass>();
}
