# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyPoissonDgTestHelpers")

spectre_python_add_module(
  DgSchemes
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "TestHelpers/Poisson"
  SOURCES
  Bindings.cpp
  FirstOrder.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainCreators
  PoissonDgTestHelpers
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyDomainCreators
  )
