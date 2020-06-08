# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Multigrid)

set(LIBRARY ParallelLinearSolver)

add_spectre_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Convergence
  DataStructures
  LinearSolver
  IO
  Utilities
  )
