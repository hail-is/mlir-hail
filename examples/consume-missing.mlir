func @consume_missing(%b: i1, %i: i32) -> i32 {
  %0 = optional.missing : !optional.option<i32, i64>
  %1 = optional.consume_opt(%0) {
    optional.yield %i: i32
  }, {
  ^bb0(%v1: i32, %v2: i64):
    optional.yield %v1: i32
  } : (!optional.option<i32, i64>) -> i32
  return %1 : i32
}
