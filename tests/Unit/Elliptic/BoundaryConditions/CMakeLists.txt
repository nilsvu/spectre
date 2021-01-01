# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticBoundaryConditions")

set(LIBRARY_SOURCES
  Test_AnalyticSolution.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/BoundaryConditions/"
  "${LIBRARY_SOURCES}"
  "DataStructures;DomainStructure;Elliptic;Utilities"
  )
