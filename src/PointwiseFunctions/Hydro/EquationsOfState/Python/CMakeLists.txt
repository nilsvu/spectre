# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyEquationsOfState")

spectre_python_add_module(
  EquationsOfState
  MODULE_PATH "PointwiseFunctions/Hydro"
  SOURCES
  Bindings.cpp
  EquationOfState.cpp
  PolytropicFluid.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  PUBLIC Hydro
  ${SPECTRE_LINK_PYBINDINGS}
  )
