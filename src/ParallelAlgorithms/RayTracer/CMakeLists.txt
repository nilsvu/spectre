# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RayTracer)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BackgroundSpacetimes/NumericData.cpp
  RaySources/Camera.cpp
  RayTracer.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BackgroundSpacetimes/BackgroundSpacetime.hpp
  BackgroundSpacetimes/Factory.hpp
  BackgroundSpacetimes/NumericData.hpp
  BackgroundSpacetimes/WrappedGr.hpp
  RaySources/Camera.hpp
  RaySources/Factory.hpp
  RaySources/RaySource.hpp
  RayTracer.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Exporter
  GeneralRelativity
  GeneralRelativitySolutions
  IO
  Logging
  Options
  Parallel
  RelativisticEulerSolutions
  Utilities
  )
