# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ApparentHorizons
  CoordinateMaps
  DomainCreators
  EllipticDg
  GeneralRelativity
  Informer
  Importers
  IO
  LinearOperators
  Options
  Parallel
  ParallelInterpolation
  Utilities
  )

add_spectre_parallel_executable(
  FindHorizons3D
  FindHorizons
  Executables/FindHorizons
  FindHorizons::Metavariables<3>
  "${LIBS_TO_LINK}"
  )
