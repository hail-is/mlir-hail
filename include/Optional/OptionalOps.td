#ifndef HAIL_OPTIONAL_OPS
#define HAIL_OPTIONAL_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "OptionalDialect.td"

def PresentOp : Optional_Op<"present", [NoSideEffect]> {
  let summary = "present";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let results = (outs OptionalType:$result);

  let assemblyFormat = [{
   `(` $values `)` attr-dict `:` functional-type($values, $result)
  }];
}

def MissingOp : Optional_Op<"missing", [NoSideEffect]> {
  let summary = "missing";
  let description = [{
  }];

  let results = (outs OptionalType:$result);

  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def ConsumeOptOp : Optional_Op<"consume_opt", [
      DeclareOpInterfaceMethods<RegionBranchOpInterface>,
      SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "consume_opt";
  let description = [{
  }];

  let arguments = (ins OptionalType:$input);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$missingRegion, SizedRegion<1>:$presentRegion);

  let verifier = [{ return ::verify(*this); }];

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` $input `)` $missingRegion `,` $presentRegion attr-dict `:` functional-type($input, results)
  }];

  let extraClassDeclaration = [{
    mlir::Block *missingBlock();
    YieldOp missingYield();
    mlir::Block *presentBlock();
    YieldOp presentYield();
  }];
}

def YieldOp : Optional_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                                    ParentOneOf<["ConsumeOptOp"]>]> {
  let summary = "consume_opt yield and termination operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{ attr-dict ($results^ `:` type($results))? }];
}

def ConstructCoOptOp : CoOptional_Op<"construct_co_opt"> {
  let summary = "construct a co-optional type";
  let description = [{
  }];

  let results = (outs CoOptionalType:$results);

  let regions = (region AnyRegion:$reg);

  let assemblyFormat =
      [{ $reg attr-dict `:` type($results) }];
}

def ConsumeCoOptOp : CoOptional_Op<"consume_co_opt", [Terminator]> {
  let summary = "consume a co-optional type";
  let description = [{
  }];

  // TODO Make $missing and $present `ContinuationType`s
  let arguments = (ins CoOptionalType:$opt, AnyType:$missing, AnyType:$present);

  let assemblyFormat = [{
    `(` $opt `,` $missing `,` $present `)` attr-dict `:` `(` type($opt) `,` type($missing) `,` type($present) `)`
  }];
}

def PackOptionalOp : Optional_Op<"pack_opt", [NoSideEffect]> {
  let summary = "pack_opt";
  let description = [{
  }];

  let arguments = (ins I1:$isPresent, AnyType:$value);

  let results = (outs OptionalType:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $result)
  }];
}

def UnpackOptionalOp : Optional_Op<"unpack_opt", [NoSideEffect]> {
  let summary = "unpack_opt";
  let description = [{
  }];

  let arguments = (ins OptionalType:$input);

  let results = (outs I1:$isDefined, AnyType:$value);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, results)
  }];
}

def UndefinedOp : Optional_Op<"undefined", [NoSideEffect]> {
  let summary = "undefined";
  let description = [{
  }];

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

#endif
