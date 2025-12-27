# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyRayTracer")

spectre_python_add_module(
  RayTracer
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  RayTracer
  pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
