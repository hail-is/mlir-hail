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

## Setting up this build

There are no targets yet so a build will do nothing, but to set up a build:

```sh
mkdir build
cd build
# same as the install directory above in the build/install LLVM
MLIR_DIR=~/src/mlir-hail/cmake .. -G Ninja
```
