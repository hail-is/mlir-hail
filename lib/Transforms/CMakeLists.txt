add_mlir_library(HailTransforms
  RequirednessPropagation.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Transforms

  DEPENDS
  HailTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRTransforms
  MLIRPass
)
