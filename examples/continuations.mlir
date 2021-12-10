func @continuations(%i1: i32, %i2: i64) -> i32 {
  %0, %1 = control.callcc : (i32, i64) {
  ^bb0(%ret: !control.cont<i32, i64>):
    %c1 = control.defcont : !control.cont<i32> {
     ^bb1(%arg: i32):
       control.apply(%ret, %arg, %i2) : (!control.cont<i32, i64>, i32, i64)
    }
    control.apply(%c1, %i1) : (!control.cont<i32>, i32)
  }
  return %0 : i32
}
