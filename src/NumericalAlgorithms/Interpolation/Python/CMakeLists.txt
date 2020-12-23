# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyInterpolation")


spectre_python_add_module(
  Interpolation
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  RegularGridInterpolant.cpp
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  Interpolation
  pybind11::module
)
