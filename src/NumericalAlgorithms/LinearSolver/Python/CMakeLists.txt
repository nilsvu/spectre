# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyLinearSolver")

spectre_python_add_module(
  LinearSolver
  MODULE_PATH "NumericalAlgorithms"
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  Gmres.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  LinearSolver
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  )