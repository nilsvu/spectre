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
  Utilities
  ${SPECTRE_LIBRARIES}
  )

function(add_linear_eliptic_executable
  EXECUTABLE_NAME ELLIPTIC_SYSTEM EXTRA_LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "Solve${EXECUTABLE_NAME}"
    SolveLinearEllipticProblem
    Elliptic/Executables/Linear
    "Metavariables<${ELLIPTIC_SYSTEM}>"
    "${LIBS_TO_LINK};${EXTRA_LIBS_TO_LINK}"
    )
endfunction(add_linear_eliptic_executable)

# Poisson system
function(add_poisson_executable DIM)
  add_linear_eliptic_executable(
    PoissonProblem${DIM}D
    Poisson::FirstOrderSystem<${DIM}>
    "Poisson;PoissonSolutions"
    )
endfunction(add_poisson_executable)
add_poisson_executable(1)
add_poisson_executable(2)
add_poisson_executable(3)
