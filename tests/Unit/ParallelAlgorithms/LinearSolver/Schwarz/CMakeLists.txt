# Distributed under the MIT License.
# See LICENSE.txt for details.

add_distributed_linear_solver_algorithm_test("DistributedSchwarzAlgorithm")

add_distributed_linear_solver_algorithm_test("SchwarzSmoothing")
target_link_libraries(
  Test_SchwarzSmoothing
  PUBLIC
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  LinearOperators
  LinearSolver
  Poisson
  PoissonSolutions
  )
