// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Elliptic/Amr/Actions/ProjectAmrIterationId.hpp"
#include "Elliptic/Amr/Actions/StopAmr.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Loehner.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Persson.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/TruncationError.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Variables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/BuildMatrix.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace SolvePoisson::OptionTags {
struct LinearSolverGroup {
  static std::string name() { return "LinearSolver"; }
  static constexpr Options::String help =
      "The iterative Krylov-subspace linear solver";
};
struct GmresGroup {
  static std::string name() { return "GMRES"; }
  static constexpr Options::String help = "Options for the GMRES linear solver";
  using group = LinearSolverGroup;
};
struct SchwarzSmootherGroup {
  static std::string name() { return "SchwarzSmoother"; }
  static constexpr Options::String help = "Options for the Schwarz smoother";
  using group = LinearSolverGroup;
};
struct MultigridGroup {
  static std::string name() { return "Multigrid"; }
  static constexpr Options::String help = "Options for the multigrid";
  using group = LinearSolverGroup;
};
}  // namespace SolvePoisson::OptionTags

/// \cond
template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using system =
      Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;

  using background_tag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;
  using initial_guess_tag =
      elliptic::Tags::InitialGuess<elliptic::analytic_data::InitialGuess>;

  static constexpr Options::String help{
      "Find the solution to a Poisson problem."};

  // These are the fields we solve for
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;
  // These are the fixed sources, i.e. the RHS of the equations
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  // This is the linear operator applied to the fields. We'll only use it to
  // apply the operator to the initial guess, so an optimization would be to
  // re-use the `operator_applied_to_vars_tag` below. This optimization needs a
  // few minor changes to the parallel linear solver algorithm.
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not guaranteed to be symmetric. It can be made symmetric by multiplying by
  // the DG mass matrix.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, fields_tag, SolvePoisson::OptionTags::LinearSolverGroup,
      true, fixed_sources_tag, LinearSolver::multigrid::Tags::IsFinestGrid>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;
  // Precondition each linear solver iteration with a multigrid V-cycle
  using multigrid = LinearSolver::multigrid::Multigrid<
      volume_dim, typename linear_solver::operand_tag,
      SolvePoisson::OptionTags::MultigridGroup, elliptic::dg::Tags::Massive,
      typename linear_solver::preconditioner_source_tag>;
  // Smooth each multigrid level with a number of Schwarz smoothing steps
  using subdomain_operator =
      elliptic::dg::subdomain_operator::SubdomainOperator<
          system, SolvePoisson::OptionTags::SchwarzSmootherGroup>;
  using schwarz_smoother = LinearSolver::Schwarz::Schwarz<
      typename multigrid::smooth_fields_tag,
      SolvePoisson::OptionTags::SchwarzSmootherGroup, subdomain_operator,
      tmpl::list<>, typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using vars_tag = typename linear_solver::operand_tag;
  using operator_applied_to_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, vars_tag>;
  // We'll buffer the corresponding fluxes in this tag, but won't actually need
  // to access them outside applying the operator
  using fluxes_vars_tag =
      ::Tags::Variables<db::wrap_tags_in<LinearSolver::Tags::Operand,
                                         typename system::primal_fluxes>>;

  using analytic_solution_fields = typename system::primal_fields;
  using error_compute = ::Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;
  using observe_fields = tmpl::append<
      analytic_solution_fields, error_tags,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>>,
      typename fixed_sources_tag::tags_list,
      typename db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                                  fields_tag>::tags_list,
      typename db::add_tag_prefix<LinearSolver::Tags::Operand,
                                  fields_tag>::tags_list>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 error_compute>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<background_tag, initial_guess_tag,
                 ::amr::Criteria::Tags::Criteria,
                 domain::Tags::RadiallyCompressedCoordinatesOptions>;

  using amr_iteration_id =
      Convergence::Tags::IterationId<::amr::OptionTags::AmrGroup>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   Poisson::Solutions::all_analytic_solutions<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   Poisson::Solutions::all_analytic_solutions<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   Poisson::Solutions::all_analytic_solutions<volume_dim>>,
        tmpl::pair<
            ::MathFunction<volume_dim, Frame::Inertial>,
            MathFunctions::all_math_functions<volume_dim, Frame::Inertial>>,
        tmpl::pair<
            elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
            Poisson::BoundaryConditions::standard_boundary_conditions<system>>,
        tmpl::pair<::amr::Criterion,
                   tmpl::list<::amr::Criteria::TruncationError<
                                  volume_dim, tmpl::list<Poisson::Tags::Field>>,
                              ::amr::Criteria::Loehner<
                                  volume_dim, tmpl::list<Poisson::Tags::Field>>,
                              ::amr::Criteria::Persson<
                                  volume_dim, tmpl::list<Poisson::Tags::Field>>,
                              ::amr::Criteria::DriveToTarget<volume_dim>>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, observer_compute_tags,
                           LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                ::amr::OptionTags::AmrGroup>>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                // Phase for building a matrix representation of the operator
                PhaseControl::VisitAndReturn<Parallel::Phase::BuildMatrix>,
                // Phases for AMR
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>>>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          linear_solver, multigrid, schwarz_smoother>>>;

  using dg_operator = elliptic::dg::Actions::DgOperator<
      system, true, linear_solver_iteration_id, vars_tag, fluxes_vars_tag,
      operator_applied_to_vars_tag>;

  using build_linear_operator_actions = typename dg_operator::apply_actions;

  using build_matrix = LinearSolver::Actions::BuildMatrix<
      linear_solver_iteration_id, vars_tag, operator_applied_to_vars_tag,
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
      LinearSolver::multigrid::Tags::IsFinestGrid>;

  // For labeling the yaml option for RandomizeVariables
  struct RandomizeInitialGuess {};

  using init_analytic_solution_action =
      elliptic::Actions::InitializeOptionalAnalyticSolution<
          volume_dim, background_tag,
          tmpl::append<typename system::primal_fields,
                       typename system::primal_fluxes>,
          elliptic::analytic_data::AnalyticSolution>;

  using initialization_actions = tmpl::list<
      elliptic::dg::Actions::InitializeDomain<volume_dim>,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename schwarz_smoother::initialize_element,
      Initialization::Actions::InitializeItems<
          ::amr::Initialization::Initialize<volume_dim>>,
      elliptic::Actions::InitializeFields<system, initial_guess_tag>,
      Actions::RandomizeVariables<
          ::Tags::Variables<typename system::primal_fields>,
          RandomizeInitialGuess>,
      elliptic::Actions::InitializeFixedSources<system, background_tag>,
      init_analytic_solution_action,
      elliptic::dg::Actions::initialize_operator<system>,
      Parallel::Actions::TerminatePhase>;

  using register_actions =
      tmpl::flatten<tmpl::list<observers::Actions::RegisterEventsWithObservers,
                               typename schwarz_smoother::register_element,
                               typename multigrid::register_element,
                               typename build_matrix::register_actions>>;

  template <typename Label>
  using smooth_actions =
      typename schwarz_smoother::template solve<build_linear_operator_actions,
                                                Label>;

  // These tags are communicated on subdomain overlaps to initialize the
  // subdomain geometry. AMR updates these tags, so we have to communicate them
  // after each AMR step.
  using subdomain_init_tags =
      tmpl::list<domain::Tags::Mesh<volume_dim>,
                 domain::Tags::Element<volume_dim>,
                 domain::Tags::NeighborMesh<volume_dim>>;

  using init_subdomain_action =
      elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
          system, background_tag, typename schwarz_smoother::options_group,
          false>;

  using solve_actions = tmpl::list<
      // Run AMR if requested
      PhaseControl::Actions::ExecutePhaseChange,
      // Apply the DG operator to the initial guess
      typename elliptic::dg::Actions::DgOperator<
          system, true, linear_solver_iteration_id, fields_tag, fluxes_vars_tag,
          operator_applied_to_fields_tag, vars_tag,
          fluxes_vars_tag>::apply_actions,
      // Modify fixed sources with boundary conditions. This must be done after
      // the fixed sources are reset during AMR.
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          system, fixed_sources_tag>,
      // Communicate subdomain geometry and reinitialize subdomain to account
      // for domain changes
      LinearSolver::Schwarz::Actions::SendOverlapFields<
          subdomain_init_tags, typename schwarz_smoother::options_group, false>,
      LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
          volume_dim, subdomain_init_tags,
          typename schwarz_smoother::options_group>,
      init_subdomain_action,
      // Krylov solve
      typename linear_solver::template solve<
          tmpl::list<
              // Multigrid preconditioning
              typename multigrid::template solve<
                  build_linear_operator_actions,
                  // Schwarz smoothing on each multigrid level
                  smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
                  smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>,
              ::LinearSolver::Actions::make_identity_if_skipped<
                  multigrid, build_linear_operator_actions>>,
          tmpl::list<>>,
      elliptic::Actions::RunEventsAndTriggers<amr_iteration_id>,
      elliptic::amr::Actions::StopAmr>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::list<register_actions, Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>,
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::BuildMatrix,
              tmpl::list<typename build_matrix::template actions<
                             build_linear_operator_actions>,
                         Parallel::Actions::TerminatePhase>>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename multigrid::options_group>>;

  template <typename Tag>
  using overlaps_tag = LinearSolver::Schwarz::Tags::Overlaps<
      Tag, volume_dim, SolvePoisson::OptionTags::SchwarzSmootherGroup>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors = tmpl::flatten<tmpl::list<
        elliptic::dg::ProjectGeometry<volume_dim>,
        typename linear_solver::amr_projectors,
        typename multigrid::amr_projectors,
        typename schwarz_smoother::amr_projectors,
        ::amr::projectors::DefaultInitialize<tmpl::append<
            tmpl::list<domain::Tags::InitialExtents<volume_dim>,
                       domain::Tags::InitialRefinementLevels<volume_dim>>,
            // Tags communicated on subdomain overlaps. No need to project these
            // during AMR because they will be communicated.
            db::wrap_tags_in<overlaps_tag, subdomain_init_tags>,
            // Tags initialized on subdomains. No need to project these during
            // AMR because they will get re-initialized after communication.
            typename init_subdomain_action::simple_tags>>,
        ::amr::projectors::ProjectVariables<volume_dim, fields_tag>,
        elliptic::Actions::InitializeFixedSources<system, background_tag>,
        init_analytic_solution_action,
        elliptic::dg::Actions::amr_projectors<system>,
        typename dg_operator::amr_projectors,
        typename build_matrix::amr_projectors,
        LinearSolver::multigrid::ProjectMultigridSections<dg_element_array>,
        elliptic::amr::Actions::ProjectAmrIterationId>>;
  };

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, register_actions>>;
  };

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<tmpl::list<
      dg_element_array, typename linear_solver::component_list,
      typename multigrid::component_list,
      typename schwarz_smoother::component_list,
      ::amr::Component<Metavariables>, observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>>>;

  static constexpr std::array<Parallel::Phase, 4> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Solve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
/// \endcond
