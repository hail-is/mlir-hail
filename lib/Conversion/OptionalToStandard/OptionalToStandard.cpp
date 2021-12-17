#include "Conversion/OptionalToStandard/OptionalToStandard.h"
#include "../PassDetail.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Optional/OptionalDialect.h"
#include "Control/ControlDialect.h"


using namespace mlir;
using namespace hail;
using namespace hail::optional;

namespace {

struct OptionalToStandardPass : public OptionalToStandardBase<OptionalToStandardPass> {
  void runOnOperation() override;
};

Value packOptional(PatternRewriter &rewriter, Location loc, OptionalType type, Value isPresent, ValueRange values) {
  assert(type.getValueTypes() == values.getTypes());
  llvm::SmallVector<Value, 1> results;
  llvm::SmallVector<Value, 2> toCast;
  toCast.reserve(values.size() + 1);
  toCast.push_back(isPresent);
  toCast.append(values.begin(), values.end());
  rewriter.createOrFold<UnrealizedConversionCastOp>(results, loc, type, toCast);
  return results[0];
}

struct LoweredOptional {
  explicit LoweredOptional(OptionalType type) {
    operands.reserve(type.getNumValueTypes() + 1);
  };
  Value isDefined() { return operands[0]; };
  ValueRange values() { return ValueRange(operands).drop_front(1); };

  llvm::SmallVector<Value, 2> operands{};
};

LoweredOptional unpackOptional(ConversionPatternRewriter &rewriter, Location loc, Value optional) {
  auto type = optional.getType().cast<OptionalType>();
  llvm::SmallVector<Type, 2> resultTypes;
  resultTypes.reserve(type.getNumValueTypes() + 1);
  LoweredOptional result{type};
  resultTypes.push_back(rewriter.getI1Type());
  resultTypes.append(type.getValueTypes().begin(), type.getValueTypes().end());
  rewriter.createOrFold<UnrealizedConversionCastOp>(result.operands, loc, resultTypes, optional);
  return result;
}

struct ConvertPresentOp : public OpConversionPattern<PresentOp> {
  using OpConversionPattern<PresentOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PresentOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto constTrue = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), true));
    Value newResult = packOptional(rewriter, op.getLoc(), op.getType(), constTrue, operands);
    rewriter.replaceOp(op, newResult);

    return success();
  }
};

struct ConvertMissingOp : public OpConversionPattern<MissingOp> {
  using OpConversionPattern<MissingOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MissingOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto constFalse = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), false));
    SmallVector<Value, 4> values;
    llvm::transform(op.getType().getValueTypes(), std::back_inserter(values), [&](Type type) {
      auto undefined = rewriter.create<UndefinedOp>(op.getLoc(), type);
      return undefined.getResult();
    });
    Value newResult = packOptional(rewriter, op.getLoc(), op.getType(), constFalse, values);
    rewriter.replaceOp(op, newResult);

    return success();
  }
};

struct ConvertConsumeOptOp : public OpConversionPattern<ConsumeOptOp> {
  using OpConversionPattern<ConsumeOptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ConsumeOptOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    LoweredOptional unpack = unpackOptional(rewriter, op.getLoc(), operands[0]);
    auto emptyBuilder = [](OpBuilder &, Location){};
    auto ifOp = rewriter.replaceOpWithNewOp<scf::IfOp>(op, op.getResultTypes(), unpack.isDefined(), emptyBuilder, emptyBuilder);
    rewriter.mergeBlocks(&op.missingRegion().front(), ifOp.elseBlock(), {});
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

struct ConvertUndefinedOp : public OpConversionPattern<UndefinedOp> {
  using OpConversionPattern<UndefinedOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(UndefinedOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // TODO make convert/ensure the result type is legal for LLVM
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, op.result().getType());
    return success();
  }
};

struct ConvertIfReturningOptional : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
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
    auto emptyBuilder = [](OpBuilder &, Location){};
    auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes, operands[0],
                                            emptyBuilder, emptyBuilder);
    rewriter.mergeBlocks(ifOp.thenBlock(), newOp.thenBlock(), {});
    rewriter.mergeBlocks(ifOp.elseBlock(), newOp.elseBlock(), {});

    // Add pack operations after the if, and compute the values to replace the old if
    SmallVector<Value, 4> repResults;
    size_t newIndex = 0;
    for (auto result : ifOp.getResults()) {
      Type resultType = result.getType();
      if (auto optResultType = resultType.dyn_cast<OptionalType>()) {
        size_t numValues = optResultType.getNumValueTypes();
        auto pack = rewriter.create<UnrealizedConversionCastOp>(ifOp->getLoc(), optResultType,
                                                    newOp.getResults().slice(newIndex, newIndex + numValues + 1));
        repResults.push_back(pack.getResult(0));
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

struct ConvertSCFYieldReturningOptional : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> newOperands;
    for (auto operand : operands) {
      Type type = operand.getType();
      if (auto optType = type.dyn_cast<OptionalType>()) {
        LoweredOptional unpack = unpackOptional(rewriter, op->getLoc(), operand);
        newOperands.push_back(unpack.isDefined());
        newOperands.append(unpack.values().begin(), unpack.values().end());
      } else {
        newOperands.push_back(operand);
      }
    }
    rewriter.startRootUpdate(op);
    op->setOperands(newOperands);
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

struct ConvertOptToCoOptOp : public OpConversionPattern<OptToCoOptOp> {
  using OpConversionPattern<OptToCoOptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(OptToCoOptOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto coOptType = op.getType();
    LoweredOptional unpack = unpackOptional(rewriter, op.getLoc(), operands[0]);
    auto newOp = rewriter.replaceOpWithNewOp<ConstructCoOptOp>(op, coOptType);
    newOp.bodyRegion().emplaceBlock();
    newOp.bodyRegion().addArgument(control::ContinuationType::get(rewriter.getContext(), {}));
    newOp.bodyRegion().addArgument(control::ContinuationType::get(rewriter.getContext(), coOptType.getValueTypes()));
    rewriter.setInsertionPointToStart(&newOp.body());

    auto ifOp = rewriter.create<control::IfOp>(op.getLoc(), unpack.isDefined());
    ifOp.thenRegion().emplaceBlock();
    ifOp.elseRegion().emplaceBlock();

    rewriter.setInsertionPointToStart(&ifOp.thenBlock());
    rewriter.create<control::ApplyContOp>(op.getLoc(), newOp.bodyRegion().getArgument(1), unpack.values());

    rewriter.setInsertionPointToStart(&ifOp.elseBlock());
    rewriter.create<control::ApplyContOp>(op.getLoc(), newOp.bodyRegion().getArgument(0), ValueRange{});

    return success();
  }
};

struct ConvertCoOptToOptOp : public OpConversionPattern<CoOptToOptOp> {
  using OpConversionPattern<CoOptToOptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(CoOptToOptOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // TODO: This is messy. Add more ergonomic builders for control ops to clean this up.
    auto constructCoOpt = operands[0].getDefiningOp<ConstructCoOptOp>();
    if (!constructCoOpt) return failure();

    auto optType = op.getType();

    llvm::SmallVector<Type, 2> resultTypes;
    resultTypes.reserve(optType.getNumValueTypes() + 1);
    resultTypes.push_back(rewriter.getI1Type());
    resultTypes.append(optType.getValueTypes().begin(), optType.getValueTypes().end());

    auto callcc = rewriter.create<control::CallCCOp>(op.getLoc(), resultTypes);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, optType, callcc.getResults());

    callcc.bodyRegion().emplaceBlock();
    callcc.bodyRegion().addArgument(control::ContinuationType::get(rewriter.getContext(), resultTypes));

    rewriter.setInsertionPointToStart(callcc.body());

    auto missingCont = rewriter.create<control::DefContOp>(op.getLoc(), control::ContinuationType::get(rewriter.getContext(), {}));
    auto presentCont = rewriter.create<control::DefContOp>(
            op.getLoc(), control::ContinuationType::get(rewriter.getContext(), optType.getValueTypes()));
    rewriter.mergeBlocks(&constructCoOpt.body(), callcc.body(), {missingCont, presentCont});

    missingCont.bodyRegion().emplaceBlock();
    presentCont.bodyRegion().emplaceBlock();
    presentCont.bodyRegion().addArguments(optType.getValueTypes());

    rewriter.setInsertionPointToStart(presentCont.body());
    llvm::SmallVector<Value, 2> results;
    results.reserve(optType.getNumValueTypes() + 1);
    auto constTrue = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), true));
    results.push_back(constTrue);
    results.append(presentCont.body()->getArguments().begin(), presentCont.body()->getArguments().end());
    rewriter.create<control::ApplyContOp>(op.getLoc(), callcc.body()->getArgument(0), results);

    rewriter.setInsertionPointToStart(missingCont.body());
    results.clear();
    auto constFalse = rewriter.create<ConstantOp>(op.getLoc(), BoolAttr::get(rewriter.getContext(), false));
    results.push_back(constFalse);
    llvm::transform(optType.getValueTypes(), std::back_inserter(results), [&](Type type) {
      auto undefined = rewriter.create<UndefinedOp>(op.getLoc(), type);
      return undefined.getResult();
    });
    rewriter.create<control::ApplyContOp>(op.getLoc(), callcc.body()->getArgument(0), results);

    rewriter.eraseOp(constructCoOpt);

    return success();
  }
};

}

void hail::populateOptionalToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertPresentOp, ConvertMissingOp, ConvertConsumeOptOp, ConvertOptToCoOptOp,
               ConvertCoOptToOptOp, ConvertIfReturningOptional,
               ConvertSCFYieldReturningOptional>(patterns.getContext());
}

void OptionalToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateOptionalToStdConversionPatterns(patterns);
  // Configure conversion to lower out .... Anything else is fine.
  ConversionTarget target(getContext());
  target.addIllegalOp<PresentOp, MissingOp, ConsumeOptOp, OptToCoOptOp, CoOptToOptOp>();
  target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp op) {
    for (auto result : op.results()) {
      if (result.getType().isa<OptionalType>()) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp op) {
    for (auto operand : op.getOperands()) {
      if (operand.getType().isa<OptionalType>()) return false;
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
