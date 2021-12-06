#ifndef HAIL_OPTIONAL_DIALECT
#define HAIL_OPTIONAL_DIALECT

include "mlir/IR/OpBase.td"

def Optional_Dialect : Dialect {
  let name = "optional";
  let cppNamespace = "::hail::optional";
  let description = [{
    TODO
  }];
}

class Optional_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Optional_Dialect, mnemonic, traits>;

def Optional_Type :
    DialectType<Optional_Dialect, CPred<"$_self.isa<OptionalType>()">,
                "Optional type">;

#endif // HAIL_OPTIONAL_DIALECT
