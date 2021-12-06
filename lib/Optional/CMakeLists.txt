add_mlir_dialect_library(MLIROptional
    OptionalDialect.cpp
    OptionalOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Optional

    DEPENDS
    MLIROptionalOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
