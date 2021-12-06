#ifndef HAIL_OPTIONAL_OPS
#define HAIL_OPTIONAL_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "OptionalDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PresentOp : Optional_Op<"present", []> {
  let summary = "present";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let results = (outs OptionalType);
}

def MissingOp : Optional_Op<"missing", [NoSideEffect]> {
  let summary = "missing";
  let description = [{
  }];

  let results = (outs OptionalType);
}

def ConsumeOptOp : Optional_Op<"consume_opt", []> {
  let summary = "consume_opt";
  let description = [{
  }];

  let arguments = (ins OptionalType:$input);
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion:$missingRegion, AnyRegion:$presentRegion);

  let hasCanonicalizer = 1;
}

def YieldOp : Optional_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                                    ParentOneOf<["ConsumeOptOp" ]>]> {
  let summary = "consume_opt yield and termination operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif
