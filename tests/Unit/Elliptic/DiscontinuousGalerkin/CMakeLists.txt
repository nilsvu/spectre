# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticDG")

set(LIBRARY_SOURCES
  Test_StrongSecondOrderInternalPenalty.cpp
  # Test_ImposeInhomogeneousBoundaryConditionsOnSource.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;DomainCreators;ErrorHandling;Utilities;Poisson"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
