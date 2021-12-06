get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        HailOptional
        )
add_llvm_executable(hail-opt hail-opt.cpp)

llvm_update_compile_flags(hail-opt)
target_link_libraries(hail-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(hail-opt)
