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
  let mnemonic = "continuation";

  let summary = "A continuation type";

  let parameters = (ins TypeParameter<"Type", "input type">:$inputType);

  let printer = [{
    $_printer << ContinuationType::getMnemonic() << '<' << getInputType() << '>';
  }];
  
  let parser = [{
    llvm::SMLoc loc = $_parser.getCurrentLocation();
    Type inputType;
    if ($_parser.parseLess() || $_parser.parseType(inputType))
      return ContinuationType();

    if (failed($_parser.parseGreater()))
      return ContinuationType();

    return ContinuationType::get($_ctxt, inputType);
  }];
}

#endif // HAIL_CONTROL_DIALECT
