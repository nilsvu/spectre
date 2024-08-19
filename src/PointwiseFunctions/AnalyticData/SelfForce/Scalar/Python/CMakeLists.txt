# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyScalarSelfForceAnalyticData")

spectre_python_add_module(
  ScalarSelfForce
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticData"
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ScalarSelfForceAnalyticData
  pybind11::module
  Utilities
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
)
