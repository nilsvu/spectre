// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/ApplyLinearOperatorToInitialFields.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/Actions/InitializeLinearOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFirstOrderOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
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
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainPreconditioners/ExplicitInverse.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Zero.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace SolvePoissonProblem {
namespace OptionTags {

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
struct PreconditionerGroup {
  static std::string name() noexcept { return "Preconditioner"; }
  static constexpr Options::String help =
      "The preconditioner for the linear solver";
  using group = LinearSolverGroup;
};
struct MultigridGroup {
  static std::string name() noexcept { return "Multigrid"; }
  static constexpr Options::String help =
      "Options for the multigrid preconditioner";
  using group = PreconditionerGroup;
};
struct SmootherGroup {
  static std::string name() noexcept { return "Smoother"; }
  static constexpr Options::String help =
      "The smoother on each multigrid level";
  using group = PreconditionerGroup;
};
struct SchwarzGroup {
  static std::string name() noexcept { return "Schwarz"; }
  static constexpr Options::String help =
      "Options for the Schwarz solver used for smoothing.";
  using group = SmootherGroup;
};
}  // namespace OptionTags
template <typename... Tags>
struct CombinedIterationId : db::ComputeTag {
  static std::string name() noexcept { return "CombinedIterationId"; }
  using type = tuples::TaggedTuple<Tags...>;
  using argument_tags = tmpl::list<Tags...>;
  static type function(
      const db::const_item_type<Tags>&... components) noexcept {
    return {components...};
  }
  template <typename Tag>
  using step_prefix = LinearSolver::Tags::OperatorAppliedTo<Tag>;
};
}  // namespace SolvePoissonProblem

/// \cond
template <typename System, typename Background, typename BoundaryConditions,
          typename InitialGuess>
struct Metavariables {
  using system = System;
  static constexpr size_t volume_dim = system::volume_dim;
  using background = Background;
  using boundary_conditions = BoundaryConditions;
  using initial_guess = InitialGuess;

  static constexpr bool has_analytic_solution =
      tt::conforms_to_v<Background, elliptic::protocols::AnalyticSolution>;

  static constexpr bool massive_operator = true;

  static constexpr Options::String help{
      "Find the solution to a Poisson problem."};

  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;

  // Only Dirichlet boundary conditions are currently supported, and they are
  // are all imposed by analytic solutions right now.
  // This will be generalized ASAP. We will also support numeric initial guesses
  // and analytic initial guesses that aren't solutions ("analytic data").
  using background_tag =
      tmpl::conditional_t<has_analytic_solution,
                          ::Tags::AnalyticSolution<background>,
                          elliptic::Tags::Background<background>>;
  using boundary_conditions_tag = tmpl::conditional_t<
      has_analytic_solution and std::is_same_v<boundary_conditions, background>,
      background_tag, ::Tags::BoundaryCondition<boundary_conditions>>;
  using initial_guess_tag = elliptic::Tags::InitialGuess<initial_guess>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not positive-definite for the first-order system.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, typename system::fields_tag,
      SolvePoissonProblem::OptionTags::GmresGroup, true,
      db::add_tag_prefix<::Tags::FixedSource, typename system::fields_tag>,
      LinearSolver::multigrid::Tags::IsFinestLevel>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using linear_operand_tag = typename linear_solver::operand_tag;
  using primal_variables =
      db::wrap_tags_in<LinearSolver::Tags::Preconditioned,
                       db::wrap_tags_in<LinearSolver::Tags::Operand,
                                        typename system::primal_fields>>;
  using auxiliary_variables =
      db::wrap_tags_in<LinearSolver::Tags::Preconditioned,
                       db::wrap_tags_in<LinearSolver::Tags::Operand,
                                        typename system::auxiliary_fields>>;
  static_assert(
      std::is_same_v<typename linear_operand_tag::tags_list,
                     tmpl::append<primal_variables, auxiliary_variables>>,
      "The primal and auxiliary variables must compose the linear operand (in "
      "the correct order)");

  using multigrid = LinearSolver::multigrid::Multigrid<
      Metavariables, linear_operand_tag,
      SolvePoissonProblem::OptionTags::MultigridGroup,
      typename linear_solver::preconditioner_source_tag>;
  using preconditioner_iteration_id =
      Convergence::Tags::IterationId<typename multigrid::options_group>;

  // Parse numerical flux parameters from the input file to store in the cache.
  using normal_dot_numerical_flux = Tags::NumericalFlux<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, primal_variables,
          auxiliary_variables>>;

  using smoother_subdomain_operator = elliptic::dg::SubdomainOperator<
      volume_dim, primal_variables, auxiliary_variables, fluxes_computer_tag,
      tmpl::list<>, typename system::sources, tmpl::list<>,
      normal_dot_numerical_flux, SolvePoissonProblem::OptionTags::SchwarzGroup,
      tmpl::list<>, massive_operator>;
  using smoother_subdomain_preconditioner =
      LinearSolver::Schwarz::subdomain_preconditioners::ExplicitInverse<
          volume_dim>;
  using smoother = LinearSolver::Schwarz::Schwarz<
      typename multigrid::smooth_fields_tag,
      SolvePoissonProblem::OptionTags::SchwarzGroup,
      smoother_subdomain_operator, smoother_subdomain_preconditioner,
      typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  using smoother_iteration_id =
      Convergence::Tags::IterationId<typename smoother::options_group>;

  using combined_iteration_id =
      SolvePoissonProblem::CombinedIterationId<linear_solver_iteration_id,
                                               preconditioner_iteration_id,
                                               smoother_iteration_id>;

  // Specify the DG boundary scheme. We use the strong first-order scheme here
  // that only requires us to compute normals dotted into the first-order
  // fluxes.
  using boundary_scheme = dg::FirstOrderScheme::FirstOrderScheme<
      volume_dim, linear_operand_tag,
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                         linear_operand_tag>,
      normal_dot_numerical_flux, combined_iteration_id, massive_operator>;

  // Collect events and triggers
  // (public for use by the Charm++ registration code)
  using system_fields = typename system::fields_tag::tags_list;
  using multigrid_fields = typename multigrid::fields_tag::tags_list;
  using observe_fields = tmpl::append<
      system_fields, db::wrap_tags_in<::Tags::FixedSource, system_fields>,
      db::wrap_tags_in<::Tags::Analytic, system_fields>,
      tmpl::list<
          LinearSolver::Schwarz::Tags::Weight<typename smoother::options_group>,
          LinearSolver::Schwarz::Tags::SummedIntrudingOverlapWeights<
              typename smoother::options_group>>,
      typename multigrid::fields_tag::tags_list,
      typename multigrid::source_tag::tags_list,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingInitial,
                       multigrid_fields>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingSource,
                       multigrid_fields>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingResult,
                       multigrid_fields>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PreSmoothingResidual,
                       multigrid_fields>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingInitial,
                       multigrid_fields>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingSource,
                       multigrid_fields>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingResult,
                       multigrid_fields>,
      db::wrap_tags_in<LinearSolver::multigrid::Tags::PostSmoothingResidual,
                       multigrid_fields>>;
  using analytic_solution_fields =
      tmpl::conditional_t<has_analytic_solution, system_fields, tmpl::list<>>;
  using events = tmpl::flatten<
      tmpl::list<dg::Events::Registrars::ObserveFields<
                     volume_dim, linear_solver_iteration_id, observe_fields,
                     analytic_solution_fields,
                     LinearSolver::multigrid::Tags::MultigridLevel>,
                 tmpl::conditional_t<
                     has_analytic_solution,
                     dg::Events::Registrars::ObserveErrorNorms<
                         linear_solver_iteration_id, analytic_solution_fields,
                         LinearSolver::multigrid::Tags::MultigridLevel>,
                     tmpl::list<>>>>;
  using triggers = tmpl::list<elliptic::Triggers::Registrars::EveryNIterations<
      linear_solver_iteration_id,
      LinearSolver::multigrid::Tags::MultigridLevel>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<background_tag, boundary_conditions_tag, initial_guess_tag,
                 fluxes_computer_tag, normal_dot_numerical_flux,
                 Tags::EventsAndTriggers<events, triggers>>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::flatten<tmpl::list<typename Event<events>::creatable_classes,
                               linear_solver, multigrid, smoother>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<volume_dim>,
      dg::Actions::InitializeInterfaces<
          system, dg::Initialization::slice_tags_to_face<>,
          dg::Initialization::slice_tags_to_exterior<>,
          dg::Initialization::face_compute_tags<>,
          dg::Initialization::exterior_compute_tags<>, false, false>,
      elliptic::Actions::InitializeFields,
      elliptic::Actions::InitializeFixedSources,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename smoother::initialize_element,
      elliptic::dg::Actions::InitializeSubdomain<
          volume_dim, typename smoother::options_group>,
      ::Initialization::Actions::AddComputeTags<tmpl::list<
          combined_iteration_id,
          LinearSolver::Schwarz::Tags::SummedIntrudingOverlapWeightsCompute<
              volume_dim, typename smoother::options_group>>>,
      tmpl::conditional_t<has_analytic_solution,
                          elliptic::Actions::InitializeAnalyticSolution<
                              background_tag, analytic_solution_fields>,
                          tmpl::list<>>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          Metavariables>,
      dg::Actions::InitializeMortars<boundary_scheme>,
      elliptic::dg::Actions::InitializeFirstOrderOperator<
          volume_dim, typename system::fluxes, typename system::sources,
          linear_operand_tag, primal_variables, auxiliary_variables>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using build_linear_operator_actions = tmpl::list<
      dg::Actions::CollectDataForFluxes<
          boundary_scheme, domain::Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<boundary_scheme>,
      Actions::MutateApply<elliptic::FirstOrderOperator<
          volume_dim, LinearSolver::Tags::OperatorAppliedTo, linear_operand_tag,
          massive_operator>>,
      elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
          linear_operand_tag, primal_variables>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      Actions::MutateApply<boundary_scheme>>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename linear_solver::register_element,
                 typename multigrid::register_element,
                 typename smoother::register_element,
                 Parallel::Actions::TerminatePhase>;

  template <typename Label>
  using smooth_actions = tmpl::list<
      build_linear_operator_actions,
      typename smoother::template solve<build_linear_operator_actions, Label>>;

  using solve_actions = tmpl::list<
      elliptic::Actions::apply_linear_operator_to_initial_fields<
          build_linear_operator_actions, typename system::fields_tag,
          linear_operand_tag, LinearSolver::multigrid::Tags::IsFinestLevel>,
      typename linear_solver::template solve<tmpl::list<
          Actions::RunEventsAndTriggers,
          // TODO: make preconditioning the identity operation if it is
          // disabled
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
  using component_list = tmpl::flatten<tmpl::list<
      dg_element_array, typename linear_solver::component_list,
      typename multigrid::component_list, typename smoother::component_list,
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
    &setup_error_handling, &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
