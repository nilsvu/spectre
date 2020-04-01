# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Python)

set(LIBRARY "Test_XctsSolutions")

set(LIBRARY_SOURCES
  Test_ConstantDensityStar.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Xcts/"
  "${LIBRARY_SOURCES}"
  "XctsSolutions;Utilities"
  )

spectre_add_python_bindings_test(
  "Unit.AnalyticSolutions.Xcts.Python.VerifySolution"
  "Test_VerifySolution.py"
  "Unit;Elliptic;PointwiseFunctions")
