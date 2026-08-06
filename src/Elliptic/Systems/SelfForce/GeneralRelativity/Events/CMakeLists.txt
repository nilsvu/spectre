# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrSelfForceEvents)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ObserveFlux.cpp
  ObserveRedshift.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ObserveFlux.hpp
  ObserveRedshift.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  Elliptic
  GrSelfForceAnalyticData
  Options
  Serialization
  Utilities
  PRIVATE
  DiscontinuousGalerkin
  LinearOperators
  )
