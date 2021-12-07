#include "Conversion/OptionalToStandard/OptionalToStandard.h"
#include "../PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"


using namespace mlir;
using namespace hail;

namespace {

struct OptionalToStandardPass : public OptionalToStandardBase<OptionalToStandardPass> {
  void runOnOperation() override;
};

}

void hail::populateOptionalToStdConversionPatterns(RewritePatternSet &patterns) {
//  patterns.add<>(patterns.getContext());
}

void OptionalToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateOptionalToStdConversionPatterns(patterns);
  // Configure conversion to lower out .... Anything else is fine.
  ConversionTarget target(getContext());
//  target.addIllegalOp<>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> hail::createLowerOptionalToSTDPass() {
  return std::make_unique<OptionalToStandardPass>();
}
