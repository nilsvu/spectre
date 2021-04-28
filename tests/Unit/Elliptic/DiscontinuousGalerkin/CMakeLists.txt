# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticDG")

set(LIBRARY_SOURCES
  Test_DgOperator.cpp
  Test_ImposeBoundaryConditions.cpp
  Test_ImposeInhomogeneousBoundaryConditionsOnSource.cpp
  Test_InitializeFirstOrderOperator.cpp
  Test_Penalty.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  AnalyticSolutions
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainCreators
  DomainStructure
  Elliptic
  EllipticDg
  ErrorHandling
  Parallel
  ParallelDg
  Poisson
  PoissonSolutions
  Spectral
  Utilities
  )

add_subdirectory(NumericalFluxes)
add_subdirectory(SubdomainOperator)
