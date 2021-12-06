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
  let mnemonic = "optional_type";
}

#endif // HAIL_OPTIONAL_DIALECT
