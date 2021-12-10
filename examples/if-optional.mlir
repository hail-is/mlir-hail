func @if_optional(%b: i1, %i1: i32, %i2: i64) -> i32 {
  %opt = scf.if %b -> !optional.option<i32, i64> {
    %missing = optional.missing : !optional.option<i32, i64>
    scf.yield %missing : !optional.option<i32, i64>
  } else {
    %present = optional.present(%i1, %i2) : (i32, i64) -> !optional.option<i32, i64>
    scf.yield %present : !optional.option<i32, i64>
  }
  %1 = optional.consume_opt(%opt) {
    %zero = constant 0 : i32
    optional.yield %zero: i32
  }, {
  ^bb0(%v1: i32, %v2: i64):
    optional.yield %v1: i32
  } : (!optional.option<i32, i64>) -> i32
  return %1 : i32
}
