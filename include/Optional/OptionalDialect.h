#ifndef HAIL_OPTIONALDIALECT_H
#define HAIL_OPTIONALDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir


namespace hail {
namespace optional {
class CoOptionalType;

namespace detail {
struct OptionalTypeStorage;
} // end namespace detail

class OptionalType : public ::mlir::Type::TypeBase<OptionalType, ::mlir::Type, detail::OptionalTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static OptionalType get(llvm::ArrayRef<mlir::Type> valueTypes) {
    assert(!valueTypes.empty() && "expected at least 1 value type");
    return OptionalType::get(valueTypes.front().getContext(), valueTypes);
  }

  static OptionalType get(mlir::MLIRContext *ctx, llvm::ArrayRef<mlir::Type> valueTypes);

  static constexpr ::llvm::StringLiteral getMnemonic() {
    return ::llvm::StringLiteral("option");
  }

  llvm::ArrayRef<mlir::Type> getValueTypes() const;

  size_t getNumValueTypes() const { return getValueTypes().size(); }

  static ::mlir::Type parse(::mlir::MLIRContext *context, ::mlir::DialectAsmParser &parser);

  void print(::mlir::DialectAsmPrinter &printer) const;

  CoOptionalType asCoOptional() const;
};

} // namespace optional
} // namespace hail

#include "Optional/OptionalOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Optional/OptionalOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Optional/OptionalOps.h.inc"

#endif
