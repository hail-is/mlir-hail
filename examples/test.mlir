func @test(%arg0: i32) -> i32 {
  %0 = "optional.missing"() {} : () -> (i32)
  return %0 : i32
}
