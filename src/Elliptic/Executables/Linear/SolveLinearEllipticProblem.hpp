// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFirstOrderOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Weighting.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

namespace SolveLinearEllipticProblem {
namespace OptionTags {

struct LinearSolverGroup {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr OptionString help = "The iterative linear solver";
};

struct LinearSolverOptions {
  using group = LinearSolverGroup;
  static std::string name() noexcept { return "GMRES"; }
  static constexpr OptionString help =
      "Options for the iterative linear solver";
};

struct PreconditionerGroup {
  static std::string name() noexcept { return "Preconditioner"; }
  static constexpr OptionString help =
      "The preconditioner for the linear solves";
};

struct PreconditionerOptions {
  using group = PreconditionerGroup;
  static std::string name() noexcept { return "Schwarz"; }
  static constexpr OptionString help = "Options for the Schwarz preconditioner";
};

}  // namespace OptionTags

template <typename... Tags>
struct CombinedIterationId : db::ComputeTag {
  static std::string name() noexcept { return "CombinedIterationId"; }
  using argument_tags = tmpl::list<Tags...>;
  static tuples::TaggedTuple<Tags...> function(
      const db::const_item_type<Tags>&... components) noexcept {
    return {components...};
  }
  template <typename Tag>
  using step_prefix = LinearSolver::Tags::OperatorAppliedTo<Tag>;
};
}  // namespace SolveLinearEllipticProblem

/// \cond
template <typename System, typename InitialGuess, typename BoundaryConditions>
struct Metavariables {
  using system = System;
  static constexpr size_t volume_dim = system::volume_dim;
  using initial_guess = InitialGuess;
  using boundary_conditions = BoundaryConditions;

  static constexpr bool use_preconditioner = true;

  static constexpr OptionString help{
      "Find the solution to a linear elliptic problem.\n"
      "Linear solver: GMRES\n"
      "Numerical flux: FirstOrderInternalPenaltyFlux"};

  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;

  // Only Dirichlet boundary conditions are currently supported, and they are
  // are all imposed by analytic solutions right now.
  // This will be generalized ASAP. We will also support numeric initial guesses
  // and analytic initial guesses that aren't solutions ("analytic data").
  using analytic_solution_tag = Tags::AnalyticSolution<boundary_conditions>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not positive-definite for the first-order system.
  using linear_solver = LinearSolver::Gmres<
      Metavariables, typename system::fields_tag,
      SolveLinearEllipticProblem::OptionTags::LinearSolverOptions,
      use_preconditioner>;
  using linear_solver_iteration_id =
      LinearSolver::Tags::IterationId<typename linear_solver::options_group>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using linear_operand_tag = std::conditional_t<
      use_preconditioner,
      db::add_tag_prefix<LinearSolver::Tags::Preconditioned,
                         db::add_tag_prefix<LinearSolver::Tags::Operand,
                                            typename system::fields_tag>>,
      db::add_tag_prefix<LinearSolver::Tags::Operand,
                         typename system::fields_tag>>;
  using primal_variables = std::conditional_t<
      use_preconditioner,
      db::wrap_tags_in<LinearSolver::Tags::Preconditioned,
                       db::wrap_tags_in<LinearSolver::Tags::Operand,
                                        typename system::primal_fields>>,
      db::wrap_tags_in<LinearSolver::Tags::Operand,
                       typename system::primal_fields>>;
  using auxiliary_variables = std::conditional_t<
      use_preconditioner,
      db::wrap_tags_in<LinearSolver::Tags::Preconditioned,
                       db::wrap_tags_in<LinearSolver::Tags::Operand,
                                        typename system::auxiliary_fields>>,
      db::wrap_tags_in<LinearSolver::Tags::Operand,
                       typename system::auxiliary_fields>>;

  // Choose a numerical flux for the DG scheme
  using normal_dot_numerical_flux = Tags::NumericalFlux<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, primal_variables,
          auxiliary_variables>>;

  // The preconditioner for the linear solver. We use a parallel Schwarz
  // smoother.
  using subdomain_operator = elliptic::dg::SubdomainOperator<
      volume_dim, primal_variables, auxiliary_variables, fluxes_computer_tag,
      tmpl::list<>, typename system::sources, tmpl::list<>,
      normal_dot_numerical_flux,
      SolveLinearEllipticProblem::OptionTags::PreconditionerOptions>;
  using subdomain_weighting =
      elliptic::dg::SubdomainOperator_detail::Weighting<volume_dim>;
  using preconditioner = LinearSolver::Schwarz<
      Metavariables, linear_operand_tag,
      SolveLinearEllipticProblem::OptionTags::PreconditionerOptions,
      subdomain_operator, subdomain_weighting,
      // Each preconditioning operation is sourced by the GMRES "operand"
      db::add_tag_prefix<LinearSolver::Tags::Operand,
                         typename system::fields_tag>>;
  using preconditioner_iteration_id =
      LinearSolver::Tags::IterationId<typename preconditioner::options_group>;

  // Specify the DG boundary scheme. We use the strong first-order scheme here
  // that only requires us to compute normals dotted into the first-order
  // fluxes.
  using combined_iteration_id = std::conditional_t<
      use_preconditioner,
      SolveLinearEllipticProblem::CombinedIterationId<
          linear_solver_iteration_id, preconditioner_iteration_id>,
      SolveLinearEllipticProblem::CombinedIterationId<
          linear_solver_iteration_id>>;
  using boundary_scheme =
      dg::FirstOrderScheme::FirstOrderScheme<volume_dim, linear_operand_tag,
                                             normal_dot_numerical_flux,
                                             combined_iteration_id>;

  // Collect events and triggers
  // (public for use by the Charm++ registration code)
  using observe_fields =
      db::get_variables_tags_list<typename system::fields_tag>;
  using analytic_solution_fields = observe_fields;
  using events =
      tmpl::list<dg::Events::Registrars::ObserveFields<
                     volume_dim, linear_solver_iteration_id, observe_fields,
                     analytic_solution_fields>,
                 dg::Events::Registrars::ObserveErrorNorms<
                     linear_solver_iteration_id, analytic_solution_fields>>;
  using triggers = tmpl::list<elliptic::Triggers::Registrars::EveryNIterations<
      linear_solver_iteration_id>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<analytic_solution_tag, fluxes_computer_tag,
                 normal_dot_numerical_flux,
                 Tags::EventsAndTriggers<events, triggers>>;

  // Collect all reduction tags for observers
  struct element_observation_type {};
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::flatten<tmpl::list<typename Event<events>::creatable_classes,
                               linear_solver, preconditioner>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<volume_dim>,
      dg::Actions::InitializeInterfaces<
          system, dg::Initialization::slice_tags_to_face<>,
          dg::Initialization::slice_tags_to_exterior<>,
          dg::Initialization::face_compute_tags<>,
          dg::Initialization::exterior_compute_tags<>, false, false>,
      typename linear_solver::initialize_element,
      std::conditional_t<use_preconditioner,
                         typename preconditioner::initialize_element,
                         tmpl::list<>>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<combined_iteration_id>>,
      elliptic::Actions::InitializeSystem,
      elliptic::Actions::InitializeAnalyticSolution<analytic_solution_tag,
                                                    analytic_solution_fields>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          Metavariables>,
      dg::Actions::InitializeMortars<boundary_scheme>,
      elliptic::dg::Actions::InitializeFirstOrderOperator<
          volume_dim, typename system::fluxes, typename system::sources,
          linear_operand_tag, primal_variables, auxiliary_variables>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using register_actions = tmpl::list<
      observers::Actions::RegisterWithObservers<observers::RegisterObservers<
          linear_solver_iteration_id, element_observation_type>>,
      typename preconditioner::register_element,
      // We prepare the linear solve here to avoid adding an extra phase. We
      // can't do that before registration because the `prepare_solve` action
      // may contribute to observers.
      typename linear_solver::prepare_solve, Parallel::Actions::TerminatePhase>;

  using build_linear_operator_actions = tmpl::list<
      dg::Actions::CollectDataForFluxes<
          boundary_scheme, domain::Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<boundary_scheme>,
      Actions::MutateApply<elliptic::FirstOrderOperator<
          volume_dim, LinearSolver::Tags::OperatorAppliedTo,
          linear_operand_tag>>,
      elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
          linear_operand_tag, primal_variables>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      Actions::MutateApply<boundary_scheme>>;

  using solve_actions = tmpl::list<
      Actions::RunEventsAndTriggers,
      LinearSolver::Actions::TerminateIfConverged<
          typename linear_solver::options_group>,
      typename linear_solver::prepare_step, build_linear_operator_actions,
      std::conditional_t<
          use_preconditioner,
          tmpl::list<typename preconditioner::prepare_solve,
                     ::Actions::RepeatUntil<
                         LinearSolver::Tags::HasConverged<
                             typename preconditioner::options_group>,
                         tmpl::list<typename preconditioner::prepare_step,
                                    build_linear_operator_actions,
                                    typename preconditioner::perform_step>>>,
          tmpl::list<>>,
      typename linear_solver::perform_step>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Initialization,
                                        initialization_actions>,
                 Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                        register_actions>,
                 Parallel::PhaseActions<Phase, Phase::Solve, solve_actions>>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename linear_solver::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  // Specify the transitions between phases.
  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
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
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
