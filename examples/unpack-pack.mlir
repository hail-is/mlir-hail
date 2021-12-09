func @unpack_pack(%b: i1, %i: i32) -> i32 {
  %0 = optional.pack_opt(%b, %i) : (i1, i32) -> !optional.option<i32>
  %b1, %i1 = optional.unpack_opt(%0) : (!optional.option<i32>) -> (i1, i32)
  std.return %i1 : i32
}
