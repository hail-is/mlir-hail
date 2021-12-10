module {
  llvm.mlir.global internal constant @even_fmt("Even value :-) (%ld)\0A\00")
  llvm.mlir.global internal constant @odd_msg("Odd value >:|\0A\00")
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.func @lrand48() -> i64
  llvm.func @srand48(i64)
  llvm.func @time(!llvm.ptr<i64>) -> i64

  func @main() {
    %one = constant 1 : i64
    %i64_null = llvm.mlir.null : !llvm.ptr<i64>
    %time = llvm.call @time(%i64_null) : (!llvm.ptr<i64>) -> i64
    llvm.call @srand48(%time) : (i64) -> ()

    %z = constant 0 : i32
    %0 = llvm.mlir.addressof @even_fmt : !llvm.ptr<!llvm.array<22 x i8>>
    %even_fmt = llvm.getelementptr %0[%z, %z] : (!llvm.ptr<!llvm.array<22 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %1 = llvm.mlir.addressof @odd_msg : !llvm.ptr<!llvm.array<15 x i8>>
    %odd_fmt = llvm.getelementptr %1[%z, %z] : (!llvm.ptr<!llvm.array<15 x i8>>, i32, i32) -> !llvm.ptr<i8>

    %oind = constant 1 : index
    %zind = constant 0 : index
    %twenty = constant 20 : index
    scf.for %iv = %zind to %twenty step %oind {
      %rand = llvm.call @lrand48() : () -> i64
      %flip_one = llvm.xor %rand, %one : i64
      %mod2 = and %flip_one, %one : i64
      %isDefined = llvm.trunc %mod2 : i64 to i1
      %opt = optional.pack_opt(%isDefined, %rand) : (i1, i64) -> !optional.option<i64>

      optional.consume_opt(%opt) {
        %11 = llvm.call @printf(%odd_fmt) : (!llvm.ptr<i8>) -> i32
      }, {
      ^bb0(%v : i64):
        %11 = llvm.call @printf(%even_fmt, %v) : (!llvm.ptr<i8>, i64) -> i32
      } : (!optional.option<i64>) -> ()
    }

    return
  }
}
