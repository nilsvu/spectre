# Distributed under the MIT License.
# See LICENSE.txt for details.

add_distributed_linear_solver_algorithm_test("DistributedMultigridAlgorithm")
target_link_libraries(
  Test_DistributedMultigridAlgorithm
  PRIVATE
  ParallelMultigrid
)

add_distributed_linear_solver_algorithm_test(
  "DistributedMultigridPreconditionedGmresAlgorithm")
target_link_libraries(
  Test_DistributedMultigridPreconditionedGmresAlgorithm
  PRIVATE
  ParallelMultigrid
)
