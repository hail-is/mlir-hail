#ifndef HAIL_CONTROL_OPS
#define HAIL_CONTROL_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "ControlDialect.td"

def DefContOp : Control_Op<"defcont", []> {
  let summary = "Define a continuation";
  let description = [{
  }];

  let results = (outs ContinuationType:$result);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = [{ attr-dict `:` type($result) $bodyRegion }];

  let extraClassDeclaration = [{
    mlir::Block *body();
  }];
}

def ApplyContOp : Control_Op<"apply", [Terminator]> {
  let summary = "Apply a continuation";
  let description = [{
  }];

  let arguments = (ins ContinuationType:$continuation, Variadic<AnyType>:$values);

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = [{
    `(` $continuation `,` $values `)` attr-dict `:` `(` type($continuation) `,` type($values) `)`
  }];
}

def CallCCOp : Control_Op<"callcc", []> {
  let summary = "Call with current continuation";
  let description = [{
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = [{
    attr-dict `:` `(` type($results) `)` $bodyRegion
  }];

  let extraClassDeclaration = [{
    mlir::Block *body();
  }];
}

#endif
