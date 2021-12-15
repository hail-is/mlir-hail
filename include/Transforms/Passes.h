#ifndef HAIL_TRANSFORMS_PASSES_H
#define HAIL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace hail {
/// Creates a pass which performs sparse conditional constant propagation over
/// nested operations.
std::unique_ptr <mlir::Pass> createRequirednessPropagationPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"
} // end namespace hail

#endif //HAIL_TRANSFORMS_PASSES_H
