get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )
set(LIBS
        ${dialect_libs}
        ${conversion_libs}

        MLIRExecutionEngine
        MLIRIR
        MLIRJitRunner
        MLIRLLVMIR
        MLIRLLVMToLLVMIRTranslation
        MLIROptLib
        MLIRTargetLLVMIRExport
        MLIRToLLVMIRTranslationRegistration

        HailOptionalToStandard
        HailOptional
        HailControl
        )
add_llvm_executable(hailc hailc.cpp)

llvm_update_compile_flags(hailc)
target_link_libraries(hailc PRIVATE ${LIBS})

mlir_check_all_link_libraries(hailc)
