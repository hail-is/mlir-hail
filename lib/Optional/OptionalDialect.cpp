#include "Optional/OptionalDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace hail::optional;

#include "Optional/OptionalOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Optional/OptionalOpsTypes.cpp.inc"

void OptionalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Optional/OptionalOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Optional/OptionalOpsTypes.cpp.inc"
      >();
  addTypes<OptionalType>();
}

namespace hail::optional::detail {
struct OptionalTypeStorage : public ::mlir::TypeStorage {
  OptionalTypeStorage (llvm::ArrayRef<Type> valueTypes)
          : valueTypes(valueTypes) { }

  /// The hash key is a tuple of the parameter types.
  using KeyTy = llvm::ArrayRef<Type>;

  bool operator==(const KeyTy &key) const { return key == valueTypes; }

  /// Define a construction method for creating a new instance of this
  /// storage.
  static OptionalTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<OptionalTypeStorage>()) OptionalTypeStorage(elementTypes);
  }

  llvm::ArrayRef<Type> valueTypes;
};
} // namespace hail::optional::detail

llvm::ArrayRef<Type> OptionalType::getValueTypes() const {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->valueTypes;
}

OptionalType OptionalType::get(mlir::MLIRContext *ctx, llvm::ArrayRef<mlir::Type> valueTypes) {
  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after are forwarded to the storage instance.
  return Base::get(ctx, valueTypes);
}

Type OptionalType::parse(MLIRContext *context, DialectAsmParser &parser) {
  if (parser.parseLess()) return Type();

  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<Type, 1> valueTypes;
  if (failed(parser.parseOptionalGreater())) {
    do {
      // Parse the current element type.
      llvm::SMLoc typeLoc = parser.getCurrentLocation();
      Type valueType;
      if (parser.parseType(valueType)) return nullptr;
      valueTypes.push_back(valueType);

      // Parse the optional: `,`
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseGreater()) return Type();
  }

  return OptionalType::get(context, valueTypes);
}

void OptionalType::print(DialectAsmPrinter &printer) const {
  printer << OptionalType::getMnemonic() << '<';
  llvm::interleaveComma(getValueTypes(), printer);
  printer << '>';
}

Type OptionalDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  MLIRContext *context = parser.getBuilder().getContext();
  {
    Type genType;
    StringRef typeTag;
    if (failed(parser.parseKeyword(&typeTag)))
      return Type();

    if (typeTag == CoOptionalType::getMnemonic()) {
      return CoOptionalType::parse(context, parser);
    }

    if (typeTag == OptionalType::getMnemonic()) {
      return OptionalType::parse(context, parser);
    }
  }
  parser.emitError(typeLoc, "unknown type in Optional dialect");
  return Type();
}

void OptionalDialect::printType(Type type, DialectAsmPrinter &printer) const {
  return ::llvm::TypeSwitch<Type, void>(type)
          .Case<CoOptionalType>([&](CoOptionalType t) { t.print(printer); })
          .Case<OptionalType>([&](OptionalType t) { t.print(printer); })
          .Default([](::mlir::Type) {
            llvm_unreachable("unexpected 'optional' type kind");
          });
}
