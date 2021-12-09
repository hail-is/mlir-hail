func @if_optional(%b: i1, %i: i32) -> i32 {
  %opt = scf.if %b -> !optional.option<i32> {
    %missing = optional.missing : !optional.option<i32>
    scf.yield %missing : !optional.option<i32>
  } else {
    %present = optional.present(%i) : (i32) -> !optional.option<i32>
    scf.yield %present : !optional.option<i32>
  }
  %1 = optional.consume_opt(%opt) {
    %zero = constant 0 : i32
    optional.yield %zero: i32
  }, {
  ^bb0(%v: i32):
    optional.yield %v: i32
  } : (!optional.option<i32>) -> i32
  return %1 : i32
}
