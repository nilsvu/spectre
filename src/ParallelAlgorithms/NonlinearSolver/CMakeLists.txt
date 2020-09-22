# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelNonlinearSolver)

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
  DataStructures
  Informer
  Initialization
  IO
  LinearSolver
  Parallel
  ParallelLinearSolver
  Utilities
  )

add_subdirectory(NewtonRaphson)
