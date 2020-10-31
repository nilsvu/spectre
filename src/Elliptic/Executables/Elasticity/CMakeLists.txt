# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ConstitutiveRelations
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elasticity
  ElasticPotentialEnergy
  Informer
  IO
  LinearOperators
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  Utilities
  )

function(add_elasticity_executable
  EXECUTABLE_NAME ELASTICITY_SYSTEM BACKGROUND BOUNDARY_CONDITIONS INITIAL_GUESS)
  add_spectre_parallel_executable(
    "SolveElastic${EXECUTABLE_NAME}"
    SolveElasticityProblem
    Elliptic/Executables/Elasticity
    "Metavariables<${ELASTICITY_SYSTEM}, ${BACKGROUND} ${BOUNDARY_CONDITIONS}, ${INITIAL_GUESS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_elasticity_executable)

# Elasticity system with BentBeam solution
add_elasticity_executable(
  BentBeam2D
  "Elasticity::FirstOrderSystem<2>"
  Elasticity::Solutions::BentBeam
  Elasticity::Solutions::BentBeam
  Elasticity::Solutions::Zero<2>
  )
target_link_libraries(
  SolveElasticBentBeam2D
  PRIVATE
  ElasticitySolutions
  )

# Elasticity system with half-space mirror solution
add_elasticity_executable(
  HalfSpaceMirror
  "Elasticity::FirstOrderSystem<3>"
  Elasticity::Solutions::HalfSpaceMirror
  Elasticity::Solutions::HalfSpaceMirror
  Elasticity::Solutions::Zero<3>
  )
target_link_libraries(
  SolveElasticHalfSpaceMirror
  PRIVATE
  ElasticitySolutions
  )
