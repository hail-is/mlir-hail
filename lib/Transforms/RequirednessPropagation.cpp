#include "PassDetail.h"
#include "mlir/Analysis/DataFlowAnalysis.h"
#include "Optional/OptionalDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "Transforms/Passes.h"

using namespace mlir;
using namespace hail::optional;

namespace {
struct RPLatticeValue {
  enum class State {
    NonOptionalType, // value of any non-optional type
    Mixed, // value might be present or missing
    Missing, // value is always missing
    Present // value is always present
  };

  RPLatticeValue(State state = State::Mixed)
          : state(state) {}

  static RPLatticeValue getPessimisticValueState(MLIRContext *context) {
    return {};
  }

  static RPLatticeValue getPessimisticValueState(Value value) {
    return {};
  }

  bool operator==(const RPLatticeValue &rhs) const {
    return state == rhs.state;
  }

  static RPLatticeValue join(const RPLatticeValue &lhs,
                             const RPLatticeValue &rhs) {
    assert((lhs == State::NonOptionalType) == (rhs == State::NonOptionalType));
    return lhs == rhs ? lhs : State::Mixed;
  }

  llvm::StringRef toString() const {
    switch (state) {
      case State::NonOptionalType:
        return "non-optional";
      case State::Mixed:
        return "mixed";
      case State::Missing:
        return "missing";
      case State::Present:
        return "present";
    }
  }

  State state;
};

struct RPAnalysis : public ForwardDataFlowAnalysis<RPLatticeValue> {
  using ForwardDataFlowAnalysis<RPLatticeValue>::ForwardDataFlowAnalysis;

  ~RPAnalysis() override = default;

  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<RPLatticeValue> *> operands) final {
    return llvm::TypeSwitch<Operation *, ChangeResult>(op)
      .Case<MissingOp>([this](MissingOp op) {
        return getLatticeElement(op.result()).join(RPLatticeValue::State::Missing);
      })
      .Case<PresentOp>([this](PresentOp op) {
        return getLatticeElement(op.result()).join(RPLatticeValue::State::Present);
      })
      // assert all other cases do not involve Optional types, and we just have to
      // set all states to NonOptionalType
      .Default([this](Operation *op) -> ChangeResult {
        ChangeResult changed = ChangeResult::NoChange;
        for (auto result : op->getResults()) {
          assert(!result.getType().isa<OptionalType>());
          changed |= getLatticeElement(result).join(RPLatticeValue::State::NonOptionalType);
        }
        for (auto &region : op->getRegions()) {
          for (auto operand: region.getArguments()) {
            assert(!operand.getType().isa<OptionalType>());
            changed |= getLatticeElement(operand).join(RPLatticeValue::State::NonOptionalType);
          }
        }
        return changed;
      });
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// RequirednessPropagation Pass
//===----------------------------------------------------------------------===//

/// Annotate op with the inferred requiredness of all results with Optional type
static void rewrite(RPAnalysis &analysis, Operation *op) {
  MLIRContext *context = op->getContext();
  llvm::SmallVector<Attribute, 4> resultAttrs;
  for (auto result: op->getResults()) {
    if (result.getType().isa<OptionalType>()) {
      auto state = analysis.lookupLatticeElement(result);
      auto stringAttr = StringAttr::get(context, state->getValue().toString());
      resultAttrs.push_back(stringAttr);
    }
  }
  if (resultAttrs.size() != 0)
    op->setAttr("result_requiredness", ArrayAttr::get(context, resultAttrs));
}

namespace {
struct RequirednessPropagation : public RequirednessPropagationBase<RequirednessPropagation> {
  void runOnOperation() override;
};
} // end anonymous namespace

void RequirednessPropagation::runOnOperation() {
  Operation *rootOp = getOperation();

  RPAnalysis analysis(rootOp->getContext());
  analysis.run(rootOp);
  for (auto &region : rootOp->getRegions()) {
    for (auto &op : region.getOps()) {
      op.walk([&analysis](Operation *nestedOp) {
        rewrite(analysis, nestedOp);
      });
    }
  }
}

std::unique_ptr<Pass> hail::createRequirednessPropagationPass() {
  return std::make_unique<RequirednessPropagation>();
}
