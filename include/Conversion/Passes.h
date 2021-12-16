#ifndef HAIL_CONVERSION_PASSES_H
#define HAIL_CONVERSION_PASSES_H

#include "Conversion/OptionalToStandard/OptionalToStandard.h"
#include "Conversion/ControlToStandard/ControlToStandard.h"

namespace hail {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace hail

#endif // HAIL_CONVERSION_PASSES_H
