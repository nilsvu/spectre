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
  Punctures
  PunctureInitialGuesses
  Utilities
  ${SPECTRE_LIBRARIES}
  )

add_spectre_parallel_executable(
  SolvePuncturesProblem3D
  SolvePuncturesProblem
  Elliptic/Executables/Punctures
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
