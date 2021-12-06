// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func @simple_constant
func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %[[RESULT:.*]] = constant 1
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = constant 1 : i32
  %1 = constant 1 : i32
  return %0, %1 : i32, i32
}
