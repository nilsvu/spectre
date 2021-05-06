# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelMultigrid")

set(LIBRARY_SOURCES
  Test_Hierarchy.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/Multigrid"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainStructure
  ParallelMultigrid
  Utilities
  )

add_distributed_linear_solver_algorithm_test("MultigridAlgorithm")
target_link_libraries(
  Test_MultigridAlgorithm
  PRIVATE
  ParallelMultigrid
  )

add_distributed_linear_solver_algorithm_test(
  "MultigridPreconditionedGmresAlgorithm")
target_link_libraries(
  Test_MultigridPreconditionedGmresAlgorithm
  PRIVATE
  ParallelMultigrid
  )
