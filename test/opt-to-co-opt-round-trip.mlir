// RUN: hail-opt %s
func @opt_to_co_opt_round_trip() {
  %0 = optional.missing : !optional.option<i64, i32, i16, i8>
  %1 = optional.opt_to_co_opt(%0 : !optional.option<i64, i32, i16, i8>)
  %2 = optional.co_opt_to_opt(%1 : !optional.co_option<i64, i32, i16, i8>)
  return
}
