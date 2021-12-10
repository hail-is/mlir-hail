#ifndef HAIL_CONTROL_OPS
#define HAIL_CONTROL_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "ControlDialect.td"

def DefContOp : Control_Op<"define_cont", []> {
  let summary = "Define a continuation";
  let description = [{
  }];

  let results = (outs ContinuationType:$result);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = [{ $bodyRegion attr-dict `:` type($result) }];

  let extraClassDeclaration = [{
    mlir::Block *body();
  }];
}

def ApplyContOp : Control_Op<"apply_cont", [Terminator]> {
  let summary = "Apply a continuation";
  let description = [{
  }];

  let arguments = (ins ContinuationType:$k, Variadic<AnyType>:$values);

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = [{
    `(` $k `,` $values `)` attr-dict `:` `(` type($k) `,` type($values) `)`
  }];
}

def CallCCOp : Control_Op<"callcc", []> {
  let summary = "Call with current continuation";
  let description = [{
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    attr-dict `:` `(` type($results) `)` $bodyRegion
  }];
}

#endif
