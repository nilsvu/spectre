# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyXctsSolutionsTestHelpers")

spectre_python_add_module(
  Xcts
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
  Xcts
  XctsSolutions
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyDomain
  PyDomainCreators
  PyXctsSolutions
  )
