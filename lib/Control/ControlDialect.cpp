
#include "Control/ControlDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace hail::control;

#include "Control/ControlOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Control/ControlOpsTypes.cpp.inc"

void ControlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Control/ControlOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Control/ControlOpsTypes.cpp.inc"
      >();
}

Type ControlDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  {
    Type genType;
    StringRef typeTag;
    if (failed(parser.parseKeyword(&typeTag)))
      return Type();
    auto parseResult = generatedTypeParser(parser.getBuilder().getContext(),
                                           parser, typeTag, genType);
    if (parseResult.hasValue())
      return genType;
  }
  parser.emitError(typeLoc, "unknown type in Control dialect");
  return Type();
}

void ControlDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os)))
    llvm_unreachable("unexpected 'control' type kind");
}
