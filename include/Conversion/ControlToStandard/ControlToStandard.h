#ifndef HAIL_CONVERSION_CONTROLTOSTANDARD_H
#define HAIL_CONVERSION_CONTROLTOSTANDARD_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace hail {
std::unique_ptr<mlir::Pass> createLowerControlToSTDPass();
} // namespace hail

#endif // HAIL_CONVERSION_CONTROLTOSTANDARD_H
