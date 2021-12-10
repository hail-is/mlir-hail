func @consume_co_opt(%return: !control.continuation<i32>) -> i32 {
  %val = constant 3 : i32

  %0 = optional.construct_co_opt {
  ^bb0(%missing: !control.continuation<i32>, %present: !control.continuation<i32>):
    control.apply_cont(%present, %val) : (!control.continuation<i32>, i32)
  } : !optional.co_option<i32>

  optional.consume_co_opt(%0, %return, %return) : (!optional.co_option<i32>, !control.continuation<i32>, !control.continuation<i32>)
}
