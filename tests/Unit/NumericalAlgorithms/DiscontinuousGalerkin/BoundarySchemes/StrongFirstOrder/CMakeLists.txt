# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DgBoundarySchemeStrongFirstOrder")

set(LIBRARY_SOURCES
  Test_StrongFirstOrder.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder"
  "${LIBRARY_SOURCES}"
  "DataStructures;DiscontinuousGalerkin;ErrorHandling;Spectral;Utilities"
  )
