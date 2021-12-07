add_mlir_conversion_library(HailOptionalToStandard
  OptionalToStandard.cpp

#  ADDITIONAL_HEADER_DIRS
#  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToStandard

  DEPENDS
  HailConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRSCF
  MLIRTransforms
  )
