# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ConstitutiveRelations
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elasticity
  ElasticPotentialEnergy
  Informer
  IO
  LinearOperators
  Options
  ParallelLinearSolver
  ParallelMultigrid
  Utilities
  )

function(add_elasticity_executable
  EXECUTABLE_NAME ELASTICITY_SYSTEM INITIAL_GUESS BOUNDARY_CONDITIONS EXTRA_LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "Solve${EXECUTABLE_NAME}"
    SolveElasticityProblem
    Elliptic/Executables/Elasticity
    "Metavariables<${ELASTICITY_SYSTEM}, ${INITIAL_GUESS}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK};${EXTRA_LIBS_TO_LINK}"
    )
endfunction(add_elasticity_executable)

# Elasticity system with BentBeam solution
add_elasticity_executable(
  ElasticBentBeam2D
  "Elasticity::FirstOrderSystem<2>"
  Elasticity::Solutions::BentBeam
  Elasticity::Solutions::BentBeam
  "ElasticitySolutions"
  )

# Elasticity system with for half space mirror solution
add_elasticity_executable(
  HalfSpaceMirror
  "Elasticity::FirstOrderSystem<3>"
  Elasticity::Solutions::HalfSpaceMirror
  Elasticity::Solutions::HalfSpaceMirror
  "ElasticitySolutions"
  )
