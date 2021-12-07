# [MLIR](https://mlir.llvm.org) + [hail](https://hail.is) = 🚀🧬?

## Building/Installing LLVM and MLIR

```sh
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
git checkout llvmorg-13.0.0  # latest stable LLVM/MLIR release

# Two notes:
#     1. -G Ninja generates a build.ninja file rather than makefiles it's not
#        required but is recommended by LLVM
#     2. The CMAKE_INSTALL_PREFIX I put here is a subdirectory of the mlir-hail
#        (this repo's) root. If you do this, add that directory to
#        .git/info/exclude and it will be like adding it to a gitignore
cmake ../llvm -G Ninja \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_INSTALL_PREFIX=~/src/mlir-hail/llvm
ninja # this will take a while
ninja install
```

## Building the mlir-hail project

To set up the build:

```sh
mkdir build
cd build
# same prefix as the install directory above in the build/install LLVM
MLIR_DIR=~/src/mlir-hail/llvm/lib/cmake/mlir cmake .. -G Ninja
```

To build:
```sh
cd build
ninja
```

## The `optional` MLIR Dialect

As first pass, we've created the `optional` dialect to process possibly optional
values. The main definitions are in the TableGen files in
[include/Optional/](include/Optional).

The `hail-opt` binary can be used to explore MLIR modules written in our
dialect. For example:

```
# CWD=build
$ bin/hail-opt --print-ir-before-all --canonicalize ../examples/consume-present.mlir
// -----// IR Dump Before Canonicalizer //----- //
module  {
  func @consume_missing(%arg0: i1, %arg1: i32) -> i32 {
    %0 = "optional.missing"() : () -> !optional.option
    %1 = "optional.consume_opt"(%0) ( {
      optional.yield %arg1 : i32
    },  {
    ^bb0(%arg2: i32):  // no predecessors
      optional.yield %arg2 : i32
    }) : (!optional.option) -> i32
    return %1 : i32
  }
}


module  {
  func @consume_missing(%arg0: i1, %arg1: i32) -> i32 {
    return %arg1 : i32
  }
}
```
