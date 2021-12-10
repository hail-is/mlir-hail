// RUN: hail-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func @consume_co_opt(%[[CONT:.*]]: !control.cont<i32>) -> i32
func @consume_co_opt(%return: !control.cont<i32>) -> i32 {
  // CHECK-NEXT: %[[VAL:.*]] = constant 3 : i32
  // CHECK-NEXT: control.apply(%[[CONT]], %[[VAL]])

  %val = constant 3 : i32

  %0 = optional.construct_co_opt {
  ^bb0(%missing: !control.cont<>, %present: !control.cont<i32>):
    control.apply(%present, %val) : (!control.cont<i32>, i32)
  } : !optional.co_option<i32>

  optional.consume_co_opt(%0, %return, %return) : (!optional.co_option<i32>, !control.cont<i32>, !control.cont<i32>)
}
