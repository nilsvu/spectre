# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrSelfForceAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ABC.cpp
  CircularOrbit.cpp
  source_convert.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ABC.hpp
  CircularOrbit.hpp
  source_convert.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  GrEffectiveSource
  ErrorHandling
  GeneralRelativity
  KerrGeodesics
  Options
  Utilities
  )
