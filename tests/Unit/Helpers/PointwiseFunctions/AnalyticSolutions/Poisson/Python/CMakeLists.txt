# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyPoissonSolutionsTestHelpers")

spectre_python_add_module(
  Poisson
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "TestHelpers/PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  VerifySolution.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainCreators
  EllipticDgTestHelpers
  Poisson
  PoissonSolutions
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyDomain
  PyDomainCreators
  PyPoissonSolutions
  )
