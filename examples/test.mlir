func @test(%arg0: i32) -> !optional.option {
  %0 = "optional.missing"() {} : () -> (!optional.option)
  return %0 : !optional.option
}
