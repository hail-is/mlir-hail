// RUN: hail-opt %s
func @consume_co_opt() -> i32 {
  %val = constant 3 : i32

  %result = control.callcc : (i32) {
  ^bb0(%return: !control.cont<i32>):
    %0 = optional.construct_co_opt {
    ^bb1(%missing: !control.cont<i32>, %present: !control.cont<i32>):
      control.apply(%present, %val) : (!control.cont<i32>, i32)
    } : !optional.co_option<i32>

    optional.consume_co_opt(%0, %return, %return) : (!optional.co_option<i32>, !control.cont<i32>, !control.cont<i32>)
  }
  return %result : i32
}
