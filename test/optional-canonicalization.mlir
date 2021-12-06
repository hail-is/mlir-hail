// RUN: hail-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func @consume_missing(%b: i1, %i: i32) -> i32
func @consume_missing(%b: i1, %i: i32) -> i32 {
  // CHECK-NEXT: %[[RESULT:.*]] = "optional.missing"() : () -> i32
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = "optional.missing"() {} : () -> (i32)
  %1 = "optional.consume_opt"(%0) ( {
    optional.yield %i: i32
  }, {
  ^bb0(%v: i32):
    optional.yield %v: i32
  }) : (i32) -> i32
  return %1 : i32
}
