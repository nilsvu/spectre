# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ConstantDensityStar.cpp
  Schwarzschild.cpp
  Vacuum.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConstantDensityStar.hpp
  Vacuum.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Utilities
  )
