#ifndef HAIL_OPTIONAL_DIALECT
#define HAIL_OPTIONAL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Optional dialect definition.
//===----------------------------------------------------------------------===//

def Optional_Dialect : Dialect {
  let name = "optional";
  let cppNamespace = "::hail::optional";
  let description = [{
    TODO
  }];
}

//===----------------------------------------------------------------------===//
// Base optional operation definition.
//===----------------------------------------------------------------------===//

class Optional_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Optional_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Optional type definition.
//===----------------------------------------------------------------------===//

class Optional_Type<string name> : TypeDef<Optional_Dialect, name>;

def OptionalType : Optional_Type<"Optional"> {
  let mnemonic = "option";

  let summary = "an optional type";

  let parameters = (ins TypeParameter<"Type", "value type">:$valueType);

  let printer = [{
    $_printer << OptionalType::getMnemonic() << '<' << getValueType() << '>';
  }];
  
  let parser = [{
    llvm::SMLoc loc = $_parser.getCurrentLocation();
    Type elementType;
    if ($_parser.parseLess() || $_parser.parseType(elementType))
      return OptionalType();

    if (failed($_parser.parseGreater()))
      return OptionalType();

    return OptionalType::get($_ctxt, elementType);
  }];
}

//===----------------------------------------------------------------------===//
// Base optional operation definition.
//===----------------------------------------------------------------------===//

class CoOptional_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Optional_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Optional type definition.
//===----------------------------------------------------------------------===//

def CoOptionalType : Optional_Type<"CoOptional"> {
  let mnemonic = "co_option";
  let summary = "a co_option type";

  let parameters = (ins TypeParameter<"Type", "value type">:$valueType);

  let printer = [{
    $_printer << CoOptionalType::getMnemonic() << '<' << getValueType() << '>';
  }];

  let parser = [{
    llvm::SMLoc loc = $_parser.getCurrentLocation();
    Type elementType;
    if ($_parser.parseLess() || $_parser.parseType(elementType))
      return CoOptionalType();

    if (failed($_parser.parseGreater()))
      return CoOptionalType();

    return CoOptionalType::get($_ctxt, elementType);
  }];
}

#endif // HAIL_OPTIONAL_DIALECT
