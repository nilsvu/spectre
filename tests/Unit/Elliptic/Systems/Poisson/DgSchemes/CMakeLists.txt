# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PoissonDgSchemes")

set(LIBRARY_SOURCES
  Test_FirstOrder.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Poisson/DgSchemes"
  "${LIBRARY_SOURCES}"
  "DataStructures;Poisson;PoissonDgTestHelpers;Utilities"
  )

spectre_add_python_bindings_test(
  "Unit.Poisson.DgSchemes.Python.FirstOrder"
  "Test_FirstOrder.py"
  "Unit;Elliptic")
