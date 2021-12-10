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

def OptionalType :
    DialectType<Optional_Dialect, CPred<"$_self.isa<OptionalType>()">,
                "Option type", "::hail::optional::OptionalType">;

//===----------------------------------------------------------------------===//
// CoOptional type definition.
//===----------------------------------------------------------------------===//

def CoOptionalType : Optional_Type<"CoOptional"> {
  let mnemonic = "co_option";
  let summary = "a co_option type";

  let parameters = (ins ArrayRefParameter<"Type", "value types">:$valueTypes);

  let printer = [{
    $_printer << CoOptionalType::getMnemonic() << '<';
    llvm::interleaveComma(getValueTypes(), $_printer);
    printer << '>';
  }];

  let parser = [{
    if ($_parser.parseLess()) return Type();

    SmallVector<Type, 1> valueTypes;
    if (failed($_parser.parseOptionalGreater())) {
      do {
        Type valueType;
        if ($_parser.parseType(valueType)) return nullptr;
        valueTypes.push_back(valueType);
      } while (succeeded($_parser.parseOptionalComma()));
      if ($_parser.parseGreater()) return Type();
    }

    return OptionalType::get(context, valueTypes);
  }];
}

#endif // HAIL_OPTIONAL_DIALECT
