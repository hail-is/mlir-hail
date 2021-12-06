
#include "Optional/OptionalDialect.h"
#include "Optional/OptionalOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
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
}

Type OptionalDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  {
    Type genType;
    auto parseResult = generatedTypeParser(parser.getBuilder().getContext(),
                                           parser, "optional_type", genType);
    if (parseResult.hasValue())
      return genType;
  }
  parser.emitError(typeLoc, "unknown type in Optional dialect");
  return Type();
}

void OptionalDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os)))
    llvm_unreachable("unexpected 'optional' type kind");
}
