// RUN: hail-opt %s
func @consume_co_opt(%return: !control.cont<i32>) -> i32 {
  %val = constant 3 : i32

  %result = control.callcc : (i32) {
    %0 = optional.construct_co_opt {
    ^bb0(%missing: !control.cont<i32>, %present: !control.cont<i32>):
      control.apply(%present, %val) : (!control.cont<i32>, i32)
    } : !optional.co_option<i32>

    optional.consume_co_opt(%0, %return, %return) : (!optional.co_option<i32>, !control.cont<i32>, !control.cont<i32>)
  }
  return %result : i32
}
