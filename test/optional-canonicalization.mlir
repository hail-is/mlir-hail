// RUN: hail-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func @consume_missing(%arg0: i1, %[[RESULT:.*]]: i32) -> i32
func @consume_missing(%b: i1, %i: i32) -> i32 {
  // CHECK-NEXT: return %[[RESULT]]

  %0 = optional.missing : !optional.option<i32>
  %1 = optional.consume_opt(%0) {
    optional.yield %i: i32
  }, {
  ^bb0(%v: i32):
    optional.yield %v: i32
  } : (!optional.option<i32>) -> i32
  return %1 : i32
}
