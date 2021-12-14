// RUN: hail-opt %s
module {
  llvm.mlir.global internal constant @say_hello("Hello, %s!\0A\00")
  llvm.mlir.global internal constant @no_name("No Name\00")
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32

  func @hello(%str : !llvm.ptr<i8>) -> i32 {
    %z = constant 0 : i32
    %0 = llvm.mlir.addressof @say_hello : !llvm.ptr<!llvm.array<12 x i8>>
    %fmt = llvm.getelementptr %0[%z, %z] : (!llvm.ptr<!llvm.array<12 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %1 = llvm.mlir.addressof @no_name : !llvm.ptr<!llvm.array<8 x i8>>
    %mname = llvm.getelementptr %1[%z, %z] : (!llvm.ptr<!llvm.array<8 x i8>>, i32, i32) -> !llvm.ptr<i8>

    %null = llvm.mlir.null : !llvm.ptr<i8>
    %m = llvm.icmp "ne" %str, %null : !llvm.ptr<i8>
    %opt = unrealized_conversion_cast %m, %str : i1, !llvm.ptr<i8> to !optional.option<!llvm.ptr<i8>>
    %print = optional.consume_opt(%opt) {
      optional.yield %mname : !llvm.ptr<i8>
    }, {
    ^bb0(%s : !llvm.ptr<i8>):
      optional.yield %s : !llvm.ptr<i8>
    }: (!optional.option<!llvm.ptr<i8>>) -> !llvm.ptr<i8>

    %2 = llvm.call @printf(%fmt, %print) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    return %2 : i32
  }
}
