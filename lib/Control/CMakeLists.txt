add_mlir_dialect_library(HailControl
    ControlDialect.cpp
    ControlOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Control

    DEPENDS
    MLIRControlOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
