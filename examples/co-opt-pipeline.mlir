// RUN: hail-opt %s
func @co_opt_pipeline(%opt: !optional.option<i32>) -> !optional.option<i32> {
  // %mapped = optional.map(%opt) : !optional.option<i32> -> !optional.option<i32> {
  // ^bb1(%val):
  //   %newval = "dummy.map_body"(%arg1) : (i32) -> i32
  //   optional.yield(%newval) : (i32)
  // }
  %coopt = optional.opt_to_co_opt(%opt: !optional.option<i32>)
  %comapped = optional.construct_co_opt {
  ^bb1(%missing: !control.cont<>, %present: !control.cont<i32>):
    %map_body = control.defcont : !control.cont<i32> {
    ^bb0(%val: i32):
      %newval = "dummy.map_body"(%val) : (i32) -> i32
      control.apply(%present, %newval) : (!control.cont<i32>, i32)
    }
    optional.consume_co_opt(%coopt, %missing, %map_body) : (!optional.co_option<i32>, !control.cont<>, !control.cont<i32>)
  } : !optional.co_option<i32>
  %mapped = optional.co_opt_to_opt(%comapped: !optional.co_option<i32>)

  // %flatmapped = optional.flatmap(%opt) : !optional.option<i32> -> !optional.option<i32> {
  // ^bb1(%val):
  //   %inner_opt = "dummy.flatmap_body"(%val) : (i32) -> !optional.option<i32>
  //   optional.yield(%inner_opt) : (!optional.option<i32>)
  // }
  %comapped2 = optional.opt_to_co_opt(%mapped: !optional.option<i32>)
  %flatmapped = optional.construct_co_opt {
  ^bb1(%missing: !control.cont<>, %present: !control.cont<i32>):
    %map_body = control.defcont : !control.cont<i32> {
    ^bb0(%val: i32):
      %inner_opt = "dummy.flatmap_body"(%val) : (i32) -> !optional.option<i32>
      %inner_co_opt = optional.opt_to_co_opt(%inner_opt: !optional.option<i32>)
      optional.consume_co_opt(%inner_co_opt, %missing, %present) : (!optional.co_option<i32>, !control.cont<>, !control.cont<i32>)
    }
    optional.consume_co_opt(%comapped2, %missing, %map_body) : (!optional.co_option<i32>, !control.cont<>, !control.cont<i32>)
  } : !optional.co_option<i32>

  %result = optional.co_opt_to_opt(%flatmapped: !optional.co_option<i32>)
  return %result : !optional.option<i32>
}
