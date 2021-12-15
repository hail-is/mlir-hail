#ifndef HAIL_TRANSFORMS_PASSES
#define HAIL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def RequirednessPropagation : Pass<"requiredness-propagation"> {
  let summary = "Requiredness Propagation";
  let description = [{
  }];
  let constructor = "hail::createRequirednessPropagationPass()";
}

#endif // HAIL_TRANSFORMS_PASSES
