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

struct ConvertPresentOp : public OpRewritePattern<PresentOp> {
  using OpRewritePattern<PresentOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PresentOp op, PatternRewriter &rewriter) const override {
    auto constTrue = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), true));
    rewriter.replaceOpWithNewOp<PackOptionalOp>(op, op.getType(), constTrue.getResult(), op.values());

    return success();
  }
};

struct ConvertMissingOp : public OpRewritePattern<MissingOp> {
  using OpRewritePattern<MissingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MissingOp op, PatternRewriter &rewriter) const override {
    auto constFalse = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), false));
    SmallVector<Value, 4> values;
    llvm::transform(op.getType().getValueTypes(), std::back_inserter(values), [&](Type type) {
      auto undefined = rewriter.create<UndefinedOp>(op.getLoc(), type);
      return undefined.getResult();
    });
    rewriter.replaceOpWithNewOp<PackOptionalOp>(op, op.getType(), constFalse.getResult(), values);

    return success();
  }
};

struct ConvertConsumeOptOp : public OpRewritePattern<ConsumeOptOp> {
  using OpRewritePattern<ConsumeOptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConsumeOptOp op,
                                PatternRewriter &rewriter) const override {
    auto unpack = rewriter.create<UnpackOptionalOp>(op.getLoc(), rewriter.getI1Type(), op.input().getType().cast<OptionalType>().getValueTypes(), op.input());
    auto ifOp = rewriter.replaceOpWithNewOp<scf::IfOp>(op, op.getResultTypes(), unpack.isDefined(), /* withElseRegion= */ true);
    rewriter.mergeBlocks(&op.missingRegion().front(), ifOp.elseBlock());
    rewriter.mergeBlocks(&op.presentRegion().front(), ifOp.thenBlock(), unpack.values());
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
                                PatternRewriter &rewriter) const override {
    // TODO make convert/ensure the result type is legal for LLVM
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, op.result().getType());
    return success();
  }
};

struct ConvertIfReturningOptional : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  void transferBody(Block *source, Block *dest, ArrayRef<OpResult> optResults,
                    ArrayRef<size_t> mapOptsToNewIndex, PatternRewriter &rewriter) const {
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
      unsigned int newIdx = mapOptsToNewIndex[i];
      auto valueTypes = result.getType().cast<OptionalType>().getValueTypes();
      auto unpack = rewriter.create<UnpackOptionalOp>(yieldOp->getLoc(), i1Type,
                                        valueTypes,
                                        yieldOp->getOperand(newIdx));
      yieldOp->setOperands(newIdx, 1, unpack.getResults());
    }
    rewriter.finalizeRootUpdate(yieldOp);
  }

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    Type i1Type = rewriter.getI1Type();
    // compute the set of optional results, and the new expanded result types
    SmallVector<OpResult, 4> optResults;
    SmallVector<size_t, 4> mapOptsToNewIndex;
    SmallVector<Type, 4> newResultTypes;
    for (auto result : ifOp.getResults()) {
      mapOptsToNewIndex.push_back(newResultTypes.size());
      Type resultType = result.getType();
      if (auto optResultType = resultType.dyn_cast<OptionalType>()) {
        TypeRange valueTypes = optResultType.getValueTypes();
        optResults.push_back(result);
        newResultTypes.push_back(i1Type);
        llvm::copy(valueTypes, std::back_inserter(newResultTypes));
      } else {
        newResultTypes.push_back(resultType);
      }
    }
    mapOptsToNewIndex.push_back(newResultTypes.size());

    // Create a replacement operation with empty then and else regions.
    auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes, ifOp.condition(),
                                            /* withElseRegion = */ true);

    // Move the bodies and update the terminators
    transferBody(ifOp.getBody(0), newOp.getBody(0), optResults, mapOptsToNewIndex, rewriter);
    transferBody(ifOp.getBody(1), newOp.getBody(1), optResults, mapOptsToNewIndex, rewriter);
    rewriter.setInsertionPoint(ifOp);

    // Add pack operations after the if, and compute the values to replace the old if
    SmallVector<Value, 4> repResults;
    size_t newIndex = 0;
    for (auto result : ifOp.getResults()) {
      Type resultType = result.getType();
      if (auto optResultType = resultType.dyn_cast<OptionalType>()) {
        size_t numValues = optResultType.getNumValueTypes();
        auto pack = rewriter.create<PackOptionalOp>(ifOp->getLoc(), optResultType,
                                                    newOp.getResults().slice(newIndex, newIndex + numValues + 1));
        repResults.push_back(pack.result());
        newIndex += numValues + 1;
      } else {
        repResults.push_back(result);
        newIndex += 1;
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
               ConvertUndefinedOp, ConvertIfReturningOptional>(patterns.getContext());
}

void OptionalToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateOptionalToStdConversionPatterns(patterns);
  // Configure conversion to lower out .... Anything else is fine.
  ConversionTarget target(getContext());
  target.addIllegalOp<PresentOp, MissingOp, ConsumeOptOp, UndefinedOp>();
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
