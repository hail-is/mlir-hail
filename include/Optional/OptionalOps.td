#ifndef HAIL_OPTIONAL_OPS
#define HAIL_OPTIONAL_OPS

include "OptionalDialect.td"

def PresentOp : Optional_Op<"present", []> {
  let summary = "present";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let results = (outs AnyType);
}

def MissingOp : Optional_Op<"missing", []> {
  let summary = "missing";
  let description = [{
  }];

  let results = (outs AnyType);
}

def ConsumeOptOp : Optional_Op<"consume_opt", []> {
  let summary = "consume_opt";
  let description = [{
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion:$presentRegion, AnyRegion:$missingRegion);
}

#endif
