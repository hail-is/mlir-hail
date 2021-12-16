// RUN: hail-opt %s
func @consume_co_opt() -> i32 {
  %val = constant 3 : i32
  %val2 = constant 0 : i32

  %0 = optional.construct_co_opt {
  ^bb1(%missing: !control.cont<>, %present: !control.cont<i32>):
    control.apply(%present, %val) : (!control.cont<i32>, i32)
  } : !optional.co_option<i32>

  %result = control.callcc : (i32) {
  ^bb0(%return: !control.cont<i32>):
    %k = control.defcont : !control.cont<> {
      control.apply(%return, %val2) : (!control.cont<i32>, i32)
    }
    optional.consume_co_opt(%0, %k, %return) : (!optional.co_option<i32>, !control.cont<>, !control.cont<i32>)
  }
  return %result : i32
}
