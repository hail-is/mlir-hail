#ifndef HAIL_CONTROL_OPS
#define HAIL_CONTROL_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ControlDialect.td"

def DefContOp : Control_Op<"defcont", [NoSideEffect]> {
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

def CallCCOp : Control_Op<"callcc", [RecursiveSideEffects]> {
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

def IfOp : Control_Op<"if",
      //[DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       [SingleBlock, NoRegionArguments, Terminator]> {
  let summary = "if-then-else operation";
  let description = [{
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let extraClassDeclaration = [{
    mlir::Block& thenBlock();
    mlir::Block& elseBlock();
  }];

  // let hasCanonicalizer = 1;
}

#endif
