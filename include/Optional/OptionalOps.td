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
      RecursiveSideEffects,
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
    mlir::Block &missingBlock();
    YieldOp missingYield();
    mlir::Block &presentBlock();
    YieldOp presentYield();
  }];
}

def YieldOp : Optional_Op<"yield", [ReturnLike, Terminator,
                                    ParentOneOf<["ConsumeOptOp"]>]> {
  let summary = "consume_opt yield and termination operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{ attr-dict ($results^ `:` type($results))? }];
}

def OptToCoOptOp : Optional_Op<"opt_to_co_opt",
    [TypesMatchWith<"optional type parameters match co-optional's",
                   "input", "result",
                   "$_self.cast<OptionalType>().asCoOptional()">,
     NoSideEffect]> {
  let summary = "convert an optional to a co-optional";
  let description = [{ TODO }];

  let arguments = (ins OptionalType:$input);
  let results = (outs CoOptionalType:$result);

  let assemblyFormat = [{ `(` $input `:` type($input) `)` attr-dict }];

  let hasCanonicalizer = 1;
}

def CoOptToOptOp : Optional_Op<"co_opt_to_opt",
    [TypesMatchWith<"optional type parameters match co-optional's",
                   "input", "result",
                   "$_self.cast<CoOptionalType>().asOptional()">,
     NoSideEffect]> {
  let summary = "convert an co-optional to an optional";
  let description = [{ TODO }];

  let arguments = (ins CoOptionalType:$input);
  let results = (outs OptionalType:$result);

  let assemblyFormat = [{ `(` $input `:` type($input) `)` attr-dict }];

  let hasCanonicalizer = 1;
}

def ConstructCoOptOp : Optional_Op<"construct_co_opt", [NoSideEffect]> {
  let summary = "construct a co-optional type";
  let description = [{
  }];

  let results = (outs CoOptionalType:$results);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    mlir::Block &body();
  }];

  let assemblyFormat =
      [{ $bodyRegion attr-dict `:` type($results) }];
}

def ConsumeCoOptOp : Optional_Op<"consume_co_opt", [Terminator]> {
  let summary = "consume a co-optional type";
  let description = [{
  }];

  // TODO Make $missing and $present `ContinuationType`s
  let arguments = (ins CoOptionalType:$opt, AnyType:$missing, AnyType:$present);

  let hasCanonicalizer = 1;

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = [{
    `(` $opt `,` $missing `,` $present `)` attr-dict `:` `(` type($opt) `,` type($missing) `,` type($present) `)`
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
