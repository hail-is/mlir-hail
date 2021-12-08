configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

find_program(LLVM_LIT "llvm-lit"
    HINTS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_BUILD_BINARY_DIR})

if(LLVM_LIT)
    message(STATUS "found llvm-lit: ${LLVM_LIT}")
    get_filename_component(LLVM_TEST_BINARY_DIR "${LLVM_LIT}" DIRECTORY)

    find_program(FILE_CHECK "FileCheck" HINTS ${LLVM_TEST_BINARY_DIR} REQUIRED)

    add_custom_target(check COMMAND "${CMAKE_COMMAND}" -E env "PATH=$ENV{PATH}:${LLVM_TEST_BINARY_DIR}"
        "${LLVM_LIT}" "${CMAKE_CURRENT_BINARY_DIR}" -v
        DEPENDS hail-opt)
else()
    message(WARNING "llvm-lit not found, disabling tests")
endif()
