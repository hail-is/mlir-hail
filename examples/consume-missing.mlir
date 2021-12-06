func @consume_missing(%b: i1, %i: i32) -> i32 {
  %0 = "optional.missing"() {} : () -> (!optional.option)
  %1 = "optional.consume_opt"(%0) ( {
    optional.yield %i: i32
  }, {
  ^bb0(%v: i32):
    optional.yield %v: i32
  }) : (!optional.option) -> i32
  return %1 : i32
}
