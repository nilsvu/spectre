# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Informer
  IO
  LinearOperators
  MathFunctions
  Xcts
  XctsSolutions
  XctsAnalyticData
  Utilities
  ${SPECTRE_LIBRARIES}
  )

add_spectre_parallel_executable(
  SolveXctsProblem3D
  SolveXctsProblem
  Elliptic/Executables/Xcts
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
