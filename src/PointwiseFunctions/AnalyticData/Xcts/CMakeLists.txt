# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AnalyticData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  LinearOperators
  Options
  Parallel
  Spectral
  Utilities
  PRIVATE
  GeneralRelativity
  )
