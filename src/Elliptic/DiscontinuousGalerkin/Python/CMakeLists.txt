# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyEllipticDg")

spectre_python_add_module(
  dg
  MODULE_PATH elliptic
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  # ElasticityOperator.cpp
  PoissonOperator.cpp
  # XctsOperator.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstitutiveRelations
  DomainStructure
  Elasticity
  Elliptic
  EllipticDg
  Poisson
  PoissonBoundaryConditions
  pybind11::module
  Utilities
  Xcts
  XctsSolutions
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyDomainCreators
  PySpectral
  )

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#   target_link_libraries(PyEllipticDg PUBLIC OpenMP::OpenMP_CXX)
# endif()
