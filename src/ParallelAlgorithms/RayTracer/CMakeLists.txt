# Distributed under the MIT License.
# See LICENSE.txt for details.

if (NOT TARGET Kokkos::kokkos)
  return()
endif()

set(LIBRARY RayTracer)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RayTracer.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  RayTracer.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Exporter
  GeneralRelativity
  GeneralRelativitySolutions
  H5
  IO
  Logging
  Options
  Parallel
  SpectreKokkos
  Utilities
  )

add_subdirectory(BackgroundSpacetimes)
add_subdirectory(RaySources)
