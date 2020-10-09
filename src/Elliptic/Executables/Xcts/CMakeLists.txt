# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elasticity
  Informer
  IO
  LinearOperators
  Options
  ParallelLinearSolver
  ParallelMultigrid
  ParallelNonlinearSolver
  ParallelSchwarz
  Poisson
  Utilities
  Xcts
  )

function(add_xcts_executable
  EXECUTABLE_NAME XCTS_SYSTEM BOUNDARY_CONDITIONS INITIAL_GUESS ANALYTIC_SOLUTION)
  add_spectre_parallel_executable(
    "Solve${EXECUTABLE_NAME}"
    SolveXctsProblem
    Elliptic/Executables/Xcts
    "Metavariables<${XCTS_SYSTEM}, ${BOUNDARY_CONDITIONS}, ${INITIAL_GUESS}, ${ANALYTIC_SOLUTION}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_xcts_executable)

# # Poisson correction system with ProductOfSinusoids solution
# add_xcts_executable(
#   PoissonCorrection1D
#   "Poisson::FirstOrderSystem<1, Poisson::Geometry::Euclidean>"
#   Poisson::Solutions::ProductOfSinusoids<1>
#   Poisson::Solutions::ProductOfSinusoids<1>
#   )
# target_link_libraries(
#   SolvePoissonCorrection1D
#   PRIVATE
#   PoissonSolutions
#   )

# # Xcts system with ConstantDensityStar solution
# add_xcts_executable(
#   ConstantDensityStar
#   "Xcts::FirstOrderSystem<Xcts::Equations::Hamiltonian, Xcts::Geometry::Flat>"
#   Xcts::Solutions::ConstantDensityStar
#   Xcts::Solutions::ConstantDensityStar
#   )
# target_link_libraries(
#   SolveConstantDensityStar
#   PRIVATE
#   XctsSolutions
#   )

# Xcts system with Schwarzschild solution in various coordinates
add_xcts_executable(
  SchwarzschildIsotropic
  "Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianAndLapse, Xcts::Geometry::Flat>"
  Xcts::Solutions::Schwarzschild<Xcts::Solutions::SchwarzschildCoordinates::Isotropic>
  # "Xcts::BoundaryConditions::AnalyticDirichlet<3, ::Tags::AnalyticSolution<Xcts::Solutions::Schwarzschild<Xcts::Solutions::SchwarzschildCoordinates::Isotropic>>, tmpl::list<Xcts::Tags::ConformalFactor<DataVector>, Xcts::Tags::LapseTimesConformalFactor<DataVector>>>"
  Xcts::BoundaryConditions::ApparentHorizon
  Xcts::Solutions::Vacuum
  )
target_link_libraries(
  SolveSchwarzschildIsotropic
  PRIVATE
  XctsSolutions
  )
add_xcts_executable(
  SchwarzschildKerrSchildIsotropic
  "Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift, Xcts::Geometry::Flat>"
  Xcts::Solutions::Schwarzschild<Xcts::Solutions::SchwarzschildCoordinates::KerrSchildIsotropic>
  # "Xcts::BoundaryConditions::AnalyticDirichlet<3, ::Tags::AnalyticSolution<Xcts::Solutions::Schwarzschild<Xcts::Solutions::SchwarzschildCoordinates::KerrSchildIsotropic>>, tmpl::list<Xcts::Tags::ConformalFactor<DataVector>, Xcts::Tags::LapseTimesConformalFactor<DataVector>, Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>>"
  Xcts::BoundaryConditions::ApparentHorizon
  Xcts::Solutions::Vacuum
  )
target_link_libraries(
  SolveSchwarzschildKerrSchildIsotropic
  PRIVATE
  XctsSolutions
  )
# add_xcts_executable(
#   SchwarzschildPainleveGullstrand
#   "Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift, Xcts::Geometry::Flat>"
#   Xcts::Solutions::Schwarzschild<Xcts::Solutions::SchwarzschildCoordinates::PainleveGullstrand>
#   Xcts::Solutions::Schwarzschild<Xcts::Solutions::SchwarzschildCoordinates::PainleveGullstrand>
#   )
# target_link_libraries(
#   SolveSchwarzschildPainleveGullstrand
#   PRIVATE
#   XctsSolutions
#   )

# Xcts system with BlackHoleBinary boundary conditions and initial guess
add_xcts_executable(
  BlackHoleBinaryConformallyFlat
  "Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift, Xcts::Geometry::Flat>"
  Xcts::AnalyticData::BlackHoleBinary<Xcts::AnalyticData::BackgroundSpacetime::FlatKerrSchildIsotropicSliced>
  Xcts::BoundaryConditions::ApparentHorizons
  Xcts::Solutions::Vacuum
  )
target_link_libraries(
  SolveBlackHoleBinaryConformallyFlat
  PRIVATE
  XctsAnalyticData
  )
