# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelAlgorithms)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Convergence
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Events)
add_subdirectory(EventsAndTriggers)
add_subdirectory(Initialization)
add_subdirectory(LinearSolver)
