func @continuations(%i1: i32, %i2: i64) -> i32 {
  %0, %1 = control.callcc : i32 {
  ^bb0(%ret: control.cont<i32, i64>):
    control.apply_cont(%ret, %i1, %i2) : (control.cont<i32, i64>, i32, i64)
  }
  return %0
}
