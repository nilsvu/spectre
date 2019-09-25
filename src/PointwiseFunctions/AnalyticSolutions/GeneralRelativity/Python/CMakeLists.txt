# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyGeneralRelativitySolutions")

spectre_python_add_module(
  GeneralRelativitySolutions
  MODULE_PATH "PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  Tov.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC GeneralRelativitySolutions
  ${SPECTRE_LINK_PYBINDINGS}
  )
