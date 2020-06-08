# Distributed under the MIT License.
# See LICENSE.txt for details.

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
