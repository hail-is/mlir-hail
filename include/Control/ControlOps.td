#ifndef HAIL_CONTROL_OPS
#define HAIL_CONTROL_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "ControlDialect.td"

def DefCont : Control_Op<"define_cont", []> {
  let summary = "Define a continuation";
  let description = [{
  }];

  let results = (outs ContinuationType:$result);

  let regions = (region AnyRegion:$contRegion);

  let assemblyFormat = [{ $contRegion attr-dict `:` type($result) }];
}

def ApplyCont : Control_Op<"apply_cont", [Terminator]> {
  let summary = "Apply a continuation";
  let description = [{
  }];

  let arguments = (ins ContinuationType:$k, Variadic<AnyType>:$values);

  let assemblyFormat = [{
    `(` $k `,` $values `)` attr-dict `:` `(` type($k) `,` type($values) `)`
  }];
}

#endif
