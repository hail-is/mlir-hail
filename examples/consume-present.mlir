func @consume_present(%i1: i32, %i2: i64) -> i32 {
  %0 = optional.present(%i1, %i2) : (i32, i64) -> !optional.option<i32, i64>
  %1 = optional.consume_opt(%0) {
    %undef = optional.undefined : i32
    optional.yield %undef: i32
  }, {
  ^bb0(%v1: i32, %v2: i64):
    optional.yield %v1: i32
  } : (!optional.option<i32, i64>) -> i32
  return %1 : i32
}
