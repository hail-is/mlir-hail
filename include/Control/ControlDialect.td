#ifndef HAIL_CONTROL_DIALECT
#define HAIL_CONTROL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Optional dialect definition.
//===----------------------------------------------------------------------===//

def Control_Dialect : Dialect {
  let name = "control";
  let cppNamespace = "::hail::control";
  let description = [{
    TODO
  }];
}

//===----------------------------------------------------------------------===//
// Base control flow operation definition.
//===----------------------------------------------------------------------===//

class Control_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Control_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Continuation type definition.
//===----------------------------------------------------------------------===//

class Cont_Type<string name> : TypeDef<Control_Dialect, name>;

def ContinuationType : Cont_Type<"Continuation"> {
  let mnemonic = "cont";

  let summary = "A continuation type";

  let parameters = (ins ArrayRefParameter<"Type", "input types">:$inputTypes);

  let printer = [{
    $_printer << ContinuationType::getMnemonic() << '<';
    llvm::interleaveComma(getInputTypes(), $_printer);
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

    return ContinuationType::get(context, valueTypes);
  }];
}

#endif // HAIL_CONTROL_DIALECT
