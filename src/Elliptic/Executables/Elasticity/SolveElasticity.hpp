// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/ApplyLinearOperatorToInitialFields.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Elasticity/Actions/InitializeSubdomain.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveVolumeIntegrals.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Elasticity/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/Zero.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

namespace SolveElasticity::OptionTags {
struct LinearSolverGroup {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr Options::String help =
      "The iterative Krylov-subspace linear solver";
};
struct GmresGroup {
  static std::string name() noexcept { return "GMRES"; }
  static constexpr Options::String help = "Options for the GMRES linear solver";
  using group = LinearSolverGroup;
};
struct SchwarzSmootherGroup {
  static std::string name() noexcept { return "SchwarzSmoother"; }
  static constexpr Options::String help = "Options for the Schwarz smoother";
  using group = LinearSolverGroup;
};
struct MultigridGroup {
  static std::string name() noexcept { return "Multigrid"; }
  static constexpr Options::String help = "Options for the multigrid";
  using group = LinearSolverGroup;
};
}  // namespace SolveElasticity::OptionTags

/// \cond
template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using system = Elasticity::FirstOrderSystem<Dim>;

  // List the possible backgrounds, i.e. the variable-independent part of the
  // equations that define the problem to solve (along with the boundary
  // conditions). We currently only have analytic solutions implemented, but
  // will add non-solution backgrounds ASAP.
  using analytic_solution_registrars = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<Dim == 2, Elasticity::Solutions::Registrars::BentBeam,
                          tmpl::list<>>,
      tmpl::conditional_t<Dim == 3,
                          Elasticity::Solutions::Registrars::HalfSpaceMirror,
                          tmpl::list<>>>>;
  using background_registrars = analytic_solution_registrars;
  using background_tag = elliptic::Tags::Background<
      Elasticity::AnalyticData::AnalyticData<Dim, background_registrars>>;

  // List the possible initial guesses
  using initial_guess_registrars =
      tmpl::append<tmpl::list<Elasticity::Solutions::Registrars::Zero<Dim>>,
                   analytic_solution_registrars>;
  using initial_guess_tag = elliptic::Tags::InitialGuess<
      ::AnalyticData<Dim, initial_guess_registrars>>;

  static constexpr Options::String help{
      "Find the solution to a linear elasticity problem."};

  // These are the fields we solve for
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;
  // These are the fixed sources, i.e. the RHS of the equations
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not guaranteed to be symmetric. It can be made symmetric by multiplying by
  // the DG mass matrix.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, fields_tag, SolveElasticity::OptionTags::GmresGroup,
      true, db::add_tag_prefix<::Tags::FixedSource, fields_tag>,
      LinearSolver::multigrid::Tags::IsFinestLevel>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;
  // Precondition each linear solver iteration with a multigrid V-cycle
  using multigrid = LinearSolver::multigrid::Multigrid<
      volume_dim, typename linear_solver::operand_tag,
      SolveElasticity::OptionTags::MultigridGroup,
      elliptic::dg::Tags::Massive,
      typename linear_solver::preconditioner_source_tag>;
  // Smooth each multigrid level with a number of Schwarz smoothing steps
  using subdomain_operator =
      elliptic::dg::subdomain_operator::SubdomainOperator<
          system, SolveElasticity::OptionTags::SchwarzSmootherGroup,
          tmpl::list<Elasticity::Tags::ConstitutiveRelation<Dim>>>;
  using schwarz_smoother = LinearSolver::Schwarz::Schwarz<
      typename multigrid::smooth_fields_tag,
      SolveElasticity::OptionTags::SchwarzSmootherGroup, subdomain_operator,
      typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using vars_tag = typename linear_solver::operand_tag;
  using operator_applied_to_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, vars_tag>;

  // Collect events and triggers
  // (public for use by the Charm++ registration code)
  using analytic_solution_fields = typename system::primal_fields;
  using observe_fields = tmpl::append<
      analytic_solution_fields,
      tmpl::list<Elasticity::Tags::Strain<volume_dim>,
                 Elasticity::Tags::PotentialEnergyDensity<volume_dim>>,
      typename linear_solver::operand_tag::tags_list,
      typename linear_solver::preconditioner_source_tag::tags_list,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingInitial,
                       typename multigrid::fields_tag::tags_list>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingSource,
                       typename multigrid::fields_tag::tags_list>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingResult,
                       typename multigrid::fields_tag::tags_list>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingResidual,
                       typename multigrid::fields_tag::tags_list>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingInitial,
                       typename multigrid::fields_tag::tags_list>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingSource,
                       typename multigrid::fields_tag::tags_list>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingResult,
                       typename multigrid::fields_tag::tags_list>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingResidual,
                       typename multigrid::fields_tag::tags_list>>;
  using events = tmpl::list<
      dg::Events::Registrars::ObserveFields<
          volume_dim, linear_solver_iteration_id, observe_fields,
          analytic_solution_fields,
          LinearSolver::multigrid::Tags::MultigridLevel>,
      dg::Events::Registrars::ObserveErrorNorms<
          linear_solver_iteration_id, analytic_solution_fields,
          LinearSolver::multigrid::Tags::IsFinestLevel>,
      dg::Events::Registrars::ObserveVolumeIntegrals<
          volume_dim, linear_solver_iteration_id,
          tmpl::list<Elasticity::Tags::PotentialEnergyDensity<volume_dim>>,
          LinearSolver::multigrid::Tags::IsFinestLevel>>;
  using triggers = tmpl::list<elliptic::Triggers::Registrars::EveryNIterations<
      linear_solver_iteration_id>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<background_tag, initial_guess_tag,
                 Tags::EventsAndTriggers<events, triggers>>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::flatten<tmpl::list<typename Event<events>::creatable_classes,
                               linear_solver, multigrid, schwarz_smoother>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox, dg::Actions::InitializeDomain<volume_dim>,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename schwarz_smoother::initialize_element,
      elliptic::Actions::InitializeFields<system, initial_guess_tag>,
      elliptic::Actions::InitializeFixedSources<system, background_tag>,
      Initialization::Actions::AddComputeTags<tmpl::list<
          Elasticity::Tags::ConstitutiveRelationReference<volume_dim,
                                                          background_tag>,
          Elasticity::Tags::StrainCompute<volume_dim>,
          Elasticity::Tags::PotentialEnergyDensityCompute<volume_dim>>>,
      elliptic::Actions::InitializeOptionalAnalyticSolution<
          background_tag,
          tmpl::append<typename system::primal_fields,
                       typename system::primal_fluxes>,
          Elasticity::Solutions::AnalyticSolution<Dim, background_registrars>>,
      elliptic::dg::Actions::initialize_operator<
          system, linear_solver_iteration_id, vars_tag,
          operator_applied_to_vars_tag>,
      ::Initialization::Actions::AddComputeTags<
          tmpl::list<domain::Tags::InterfaceCompute<
                         domain::Tags::InternalDirections<volume_dim>,
                         domain::Tags::BoundaryCoordinates<volume_dim>>,
                     domain::Tags::InterfaceCompute<
                         domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                         domain::Tags::BoundaryCoordinates<volume_dim>>>>,
      elliptic::dg::Actions::InitializeSubdomain<
          volume_dim, typename schwarz_smoother::options_group>,
      Elasticity::Actions::InitializeSubdomain<
          volume_dim, typename schwarz_smoother::options_group>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          system, fixed_sources_tag>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using build_linear_operator_actions = elliptic::dg::Actions::apply_operator<
      system, true, linear_solver_iteration_id, vars_tag,
      operator_applied_to_vars_tag>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename multigrid::register_element,
                 typename schwarz_smoother::register_element,
                 Parallel::Actions::TerminatePhase>;

  template <typename Label>
  using smooth_actions = tmpl::list<build_linear_operator_actions,
                                    typename schwarz_smoother::template solve<
                                        build_linear_operator_actions, Label>>;

  using solve_actions = tmpl::list<
      elliptic::Actions::apply_linear_operator_to_initial_fields<
          build_linear_operator_actions, fields_tag, vars_tag>,
      typename linear_solver::template solve<tmpl::list<
          Actions::RunEventsAndTriggers,
          typename multigrid::template solve<
              smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
              smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>>>,
      Actions::RunEventsAndTriggers, Parallel::Actions::TerminatePhase>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Initialization,
                                        initialization_actions>,
                 Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                        register_actions>,
                 Parallel::PhaseActions<Phase, Phase::Solve, solve_actions>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename multigrid::options_group>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename linear_solver::component_list,
                 typename multigrid::component_list,
                 typename schwarz_smoother::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  // Specify the transitions between phases.
  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Solve;
      case Phase::Solve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        metavariables::background_tag::type::element_type>,
    &Parallel::register_derived_classes_with_charm<
        metavariables::initial_guess_tag::type::element_type>,
    &Parallel::register_derived_classes_with_charm<
        metavariables::system::boundary_conditions_base>,
    &Parallel::register_derived_classes_with_charm<
        metavariables::schwarz_smoother::subdomain_solver>,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
