# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ElasticityAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Mirror.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Mirror.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  PUBLIC DomainStructure
  PUBLIC ErrorHandling
  PUBLIC Utilities
  )
