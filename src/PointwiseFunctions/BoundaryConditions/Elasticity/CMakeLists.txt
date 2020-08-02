# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ElasticityBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  LaserBeam.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  LaserBeam.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  Elasticity
  ErrorHandling
  Options
  Utilities
  )
