# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elliptic
  EllipticDg
  Events
  Informer
  IO
  LinearOperators
  MathFunctions
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelNonlinearSolver
  ParallelSchwarz
  Xcts
  XctsAnalyticData
  XctsSolutions
  Utilities
  )

add_spectre_parallel_executable(
  SolveXcts
  SolveXcts
  Elliptic/Executables/Xcts
  "Metavariables"
  "${LIBS_TO_LINK}"
  )
