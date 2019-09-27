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
  Poisson
  PoissonSolutions
  Utilities
  ${SPECTRE_LIBRARIES}
  )

function(add_nonlinear_elliptic_executable
  EXECUTABLE_NAME ELLIPTIC_SYSTEM INITIAL_GUESS EXTRA_LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "Solve${EXECUTABLE_NAME}"
    SolveNonlinearEllipticProblem
    Elliptic/Executables/Nonlinear
    "Metavariables<${ELLIPTIC_SYSTEM}, ${INITIAL_GUESS}, Xcts::InterpolationTargetTags::StarCenter>"
    "${LIBS_TO_LINK};${EXTRA_LIBS_TO_LINK}"
    )
endfunction(add_nonlinear_elliptic_executable)

add_nonlinear_elliptic_executable(
  PoissonCorrectionProblem1D
  Poisson::FirstOrderCorrectionSystem<1>
  Poisson::Solutions::ProductOfSinusoids<1>
  "Poisson;PoissonSolutions"
  )

add_nonlinear_elliptic_executable(
  PoissonCorrectionProblem2D
  Poisson::FirstOrderCorrectionSystem<2>
  Poisson::Solutions::ProductOfSinusoids<2>
  "Poisson;PoissonSolutions"
  )

add_nonlinear_elliptic_executable(
  PoissonCorrectionProblem3D
  Poisson::FirstOrderCorrectionSystem<3>
  Poisson::Solutions::ProductOfSinusoids<3>
  "Poisson;PoissonSolutions"
  )

add_nonlinear_elliptic_executable(
  ConstantDensityStar
  "Xcts::FirstOrderSystem<3, Xcts::Equations::Hamiltonian>"
  Xcts::Solutions::ConstantDensityStar
  "Xcts;XctsSolutions"
  )

add_nonlinear_elliptic_executable(
  TovStar
  "Xcts::FirstOrderSystem<3, Xcts::Equations::HamiltonianAndLapse>"
  Xcts::Solutions::TovStar
  "Interpolation;Xcts;XctsSolutions"
  )

add_nonlinear_elliptic_executable(
  NeutronStarHeadOnCollision
  "Xcts::FirstOrderSystem<3, Xcts::Equations::HamiltonianAndLapse>"
  Xcts::AnalyticData::NeutronStarHeadOnCollision
  "Xcts;XctsAnalyticData"
  )
