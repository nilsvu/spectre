# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrSelfForceAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CircularOrbit.cpp
  CircularOrbitCoeffs.cpp
  CircularOrbitConvertEffsource.cpp
  NumericData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CircularOrbit.hpp
  CircularOrbitCoeffs.hpp
  CircularOrbitConvertEffsource.hpp
  NumericData.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  H5
  DataStructures
  ErrorHandling
  GeneralRelativity
  # GravitationalEffectiveSource
  ComovingEffectiveSource
  Options
  Utilities
  Interpolation
  )
