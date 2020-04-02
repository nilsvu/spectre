# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PyXctsSolutions)

spectre_python_add_module(
  Xcts
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  ConstantDensityStar.cpp
  Schwarzschild.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  XctsSolutions
  )
