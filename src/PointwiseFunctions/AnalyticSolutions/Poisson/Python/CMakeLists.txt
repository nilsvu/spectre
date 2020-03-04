# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PyPoissonSolutions)

spectre_python_add_module(
  Poisson
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  ProductOfSinusoids.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PUBLIC PoissonSolutions
  )
