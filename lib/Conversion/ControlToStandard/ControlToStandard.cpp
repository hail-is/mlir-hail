#include "Conversion/ControlToStandard/ControlToStandard.h"
#include "../PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Control/ControlDialect.h"

#include <vector>

using namespace mlir;
using namespace hail;
using namespace hail::control;

namespace {

struct ControlToStandardPass : public ControlToStandardBase<ControlToStandardPass> {
  void runOnOperation() override;
};

}

void ControlToStandardPass::runOnOperation() {
  std::vector<Operation *> worklist;
  worklist.reserve(64);

  // add nested ops to worklist in postorder
  for (auto &region : getOperation()->getRegions())
    region.walk([&worklist](Operation *op) {
      if (isa<CallCCOp>(op) || isa<DefContOp>(op) || isa<IfOp>(op))
        worklist.push_back(op);
    });

  for (auto *op : worklist) {
    IRRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    llvm::TypeSwitch<Operation *, void>(op)
      .Case<CallCCOp>([&, this](CallCCOp op) {
        // Split the current block before the CallCCOp to create the continuation point.
        Block *currentBlock = op->getBlock();
        Block *continuation = currentBlock->splitBlock(op);

        // add args to continuation block for each return value of op
        llvm::SmallVector<Location, 4> locs(op->getNumResults(), op.getLoc());
        continuation->addArguments(op->getResultTypes(), locs);

        // replace all uses of captured continuation (which must be ApplyContOps) with branches to the continuation block
        // make_early_inc_range finds the next user before executing the body, so that deleting the
        // current user is safe
        for (Operation *user : llvm::make_early_inc_range(op.bodyRegion().getArgument(0).getUsers())) {
          auto applyOp = dyn_cast<ApplyContOp>(user);
          if (!applyOp) signalPassFailure();
          rewriter.setInsertionPoint(applyOp);
          rewriter.replaceOpWithNewOp<BranchOp>(applyOp, applyOp.values(), continuation);
        }
        assert(op.bodyRegion().getArgument(0).use_empty() && "should be no uses of callcc body arg");
        // continuation arg now has no uses, safe to delete
        op.bodyRegion().eraseArgument(0);
        assert(op.bodyRegion().getNumArguments() == 0);

        // append the callcc body to the end of currentBlock
        Block &bodyEntryBlock = op.bodyRegion().front();
        rewriter.inlineRegionBefore(op.bodyRegion(), continuation);
        rewriter.mergeBlocks(&bodyEntryBlock, currentBlock);
        assert(op.bodyRegion().empty());

        // finally, replace results of callcc with continuation args
        rewriter.replaceOp(op, continuation->getArguments());
      })
      .Case<DefContOp>([&, this](DefContOp op) {
        Block *body = op.body();
        // inline body into parent region
        rewriter.inlineRegionBefore(op.bodyRegion(), *op->getParentRegion(), std::next(op->getBlock()->getIterator()));
        // replace all uses (which must be ApplyContOps) with branches to the body block
        // make_early_inc_range finds the next user before executing the body, so that deleting the
        // current user is safe
        for (Operation *user : llvm::make_early_inc_range(op.result().getUsers())) {
          auto applyOp = dyn_cast<ApplyContOp>(user);
          if (!applyOp) {
            applyOp->emitOpError("invalid use of continuation value");
            signalPassFailure();
          }
          rewriter.setInsertionPoint(applyOp);
          assert(body->getNumArguments() == applyOp.values().size());
          rewriter.replaceOpWithNewOp<BranchOp>(applyOp, applyOp.values(), body);
        }
        assert(op.result().use_empty() && "should be no uses of defcont result");
        // there are no more uses, safe to delete
        rewriter.eraseOp(op);
      })
      .Case<IfOp>([&, this](IfOp op) {
        Block *thenBlock = &op.thenBlock();
        Block *elseBlock = &op.elseBlock();
        rewriter.inlineRegionBefore(op.thenRegion(), *op->getParentRegion(), std::next(op->getBlock()->getIterator()));
        rewriter.inlineRegionBefore(op.elseRegion(), *op->getParentRegion(), std::next(op->getBlock()->getIterator()));
        rewriter.replaceOpWithNewOp<CondBranchOp>(op, op.condition(), thenBlock, elseBlock);
      })
      .Default([](Operation *op) {});
  }
}

std::unique_ptr<Pass> hail::createLowerControlToSTDPass() {
  return std::make_unique<ControlToStandardPass>();
}
