// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/MirrorVariables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeNonlinearSystem.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFluxes.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderCorrectionSystem.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication2.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/PopulateBoundaryMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Globalization/LineSearch/LineSearch.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/NeutronStarHeadOnCollision.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace {

struct NonlinearNumericalFluxGroup {
  static std::string name() noexcept { return "NonlinNumericalFlux"; }
  static constexpr OptionString help =
      "The numerical flux scheme for the nonlinear fields";
  using group = OptionTags::NumericalFluxGroup;
};

template <typename NumericalFluxType>
struct NonlinearNumericalFluxOption {
  static std::string name() noexcept {
    return option_name<NumericalFluxType>();
  }
  static constexpr OptionString help =
      "Options for the nonlinear numerical flux";
  using type = NumericalFluxType;
  using group = NonlinearNumericalFluxGroup;
};

template <typename NumericalFluxType>
struct NonlinearNumericalFluxTag : db::SimpleTag {
  static std::string name() noexcept { return "NonlinearNumericalFlux"; }
  using type = NumericalFluxType;
  using option_tags =
      tmpl::list<NonlinearNumericalFluxOption<NumericalFluxType>>;
  static NumericalFluxType create_from_options(
      const NumericalFluxType& numerical_flux) noexcept {
    return numerical_flux;
  }
};

}  // namespace

template <typename System, typename InitialGuess>
struct Metavariables {
  static constexpr size_t volume_dim = System::volume_dim;

  static constexpr OptionString help{
      "Find the solution to a nonlinear elliptic problem in Dim spatial "
      "dimensions.\n"
      "Elliptic system: Poisson"
      "Analytic solution: ProductOfSinusoids\n"
      "Nonlinear solver: NewtonRaphson\n"
      "Linear solver: GMRES\n"
      "Numerical flux: FirstOrderInternalPenaltyFlux"};

  // The system provides all equations specific to the problem.
  using system = System;
  using nonlinear_fields_tag = typename system::fields_tag;
  using nonlinear_fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;

  using linearized_system = typename system::linearized_system;
  using linear_fields_tag = typename linearized_system::fields_tag;
  using linear_fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename linearized_system::fluxes>;

  // Specify the analytic solution and corresponding source to define the
  // Poisson problem
  using analytic_solution_tag = Tags::AnalyticSolution<InitialGuess>;

  // Use the analytic solution to provide an initial guess for the correction
  // scheme
  using initial_guess_tag = analytic_solution_tag;

  // Specify the linear solver algorithm. We must use GMRES since the operator
  // is not positive-definite for the first-order system.
  using linear_solver = LinearSolver::Gmres<Metavariables, linear_fields_tag>;
  using linear_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, linear_fields_tag>;

  // Specify the nonlinear solver algorithm for the correction scheme
  using nonlinear_solver = NonlinearSolver::NewtonRaphson<
      Metavariables, nonlinear_fields_tag,
      NonlinearSolver::Globalization::LineSearch>;

  // Specify the DG boundary scheme. We use the strong first-order scheme here
  // that only requires us to compute normals dotted into the first-order
  // fluxes.
  using nonlinear_numerical_flux_tag = NonlinearNumericalFluxTag<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, nonlinear_fluxes_computer_tag,
          typename system::primal_variables,
          typename system::auxiliary_variables>>;
  using nonlinear_boundary_scheme =
      dg::BoundarySchemes::StrongFirstOrder<volume_dim, nonlinear_fields_tag,
                                            nonlinear_numerical_flux_tag,
                                            NonlinearSolver::Tags::IterationId>;
  using linear_numerical_flux_tag = Tags::NumericalFlux<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, linear_fluxes_computer_tag,
          typename linearized_system::primal_variables,
          typename linearized_system::auxiliary_variables>>;
  using linear_boundary_scheme =
      dg::BoundarySchemes::StrongFirstOrder<volume_dim, linear_operand_tag,
                                            linear_numerical_flux_tag,
                                            LinearSolver::Tags::IterationId>;

  // Needed by ImposeInhomogeneousBoundaryConditionsOnSource.hpp (remove ASAP)
  using normal_dot_numerical_flux = nonlinear_numerical_flux_tag;

  // Collect events and triggers
  // (public for use by the Charm++ registration code)
  using all_fields = db::get_variables_tags_list<typename system::fields_tag>;
  using observe_fields =
      tmpl::append<all_fields, db::wrap_tags_in<::Tags::Source, all_fields>>;
  using analytic_solution_fields = all_fields;
  using events = tmpl::list<
      dg::Events::Registrars::ObserveFields<
          volume_dim, NonlinearSolver::Tags::IterationId, observe_fields,
          analytic_solution_fields>,
      dg::Events::Registrars::ObserveErrorNorms<
          NonlinearSolver::Tags::IterationId, analytic_solution_fields>>;
  using triggers = tmpl::list<elliptic::Triggers::Registrars::EveryNIterations<
      NonlinearSolver::Tags::IterationId>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<nonlinear_fluxes_computer_tag, linear_fluxes_computer_tag,
                 nonlinear_numerical_flux_tag, linear_numerical_flux_tag,
                 Tags::EventsAndTriggers<events, triggers>>;

  // Collect all reduction tags for observers
  struct element_observation_type {};
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::flatten<tmpl::list<typename Event<events>::creatable_classes,
                               linear_solver, nonlinear_solver>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  // Construct the DgElementArray parallel component
  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<volume_dim>,
      Initialization::Actions::AddComputeTags<
          typename InitialGuess::compute_tags>,
      elliptic::Actions::InitializeNonlinearSystem,
      elliptic::Actions::InitializeAnalyticSolution<analytic_solution_tag,
                                                    analytic_solution_fields>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<nonlinear_fields_tag,
                                                 linear_operand_tag>,
          dg::Initialization::slice_tags_to_exterior<>,
          dg::Initialization::face_compute_tags<>,
          dg::Initialization::exterior_compute_tags<
              // We mirror the system variables to the exterior (ghost)
              // faces to impose homogeneous (zero) boundary conditions.
              // Non-zero boundary conditions are handled as contributions
              // to the source term during initialization.
              ::Tags::MirrorVariables<
                  volume_dim, ::Tags::BoundaryDirectionsInterior<volume_dim>,
                  nonlinear_fields_tag, typename system::primal_fields>,
              ::Tags::MirrorVariables<
                  volume_dim, ::Tags::BoundaryDirectionsInterior<volume_dim>,
                  linear_operand_tag,
                  typename linearized_system::primal_variables>>,
          false>,
      elliptic::dg::Actions::InitializeFluxes<system>,
      elliptic::dg::Actions::InitializeFluxes<linearized_system>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          Metavariables>,
      typename nonlinear_solver::initialize_element,
      typename linear_solver::initialize_element,
      dg::Actions::InitializeMortars<linear_boundary_scheme>,
      dg::Actions::InitializeMortars<nonlinear_boundary_scheme>,
      Parallel::Actions::TerminatePhase>;

  using build_nonlinear_operator =
      tmpl::list<dg::Actions::SendDataForFluxes<nonlinear_boundary_scheme>,
                 Actions::MutateApply<elliptic::FirstOrderOperator<
                     volume_dim, NonlinearSolver::Tags::OperatorAppliedTo,
                     nonlinear_fields_tag>>,
                 Actions::MutateApply<
                     ::dg::PopulateBoundaryMortars<nonlinear_boundary_scheme>>,
                 dg::Actions::ReceiveDataForFluxes<nonlinear_boundary_scheme>,
                 Actions::MutateApply<nonlinear_boundary_scheme>>;

  using build_linear_operator =
      tmpl::list<dg::Actions::SendDataForFluxes<linear_boundary_scheme>,
                 Actions::MutateApply<elliptic::FirstOrderOperator<
                     volume_dim, LinearSolver::Tags::OperatorAppliedTo,
                     linear_operand_tag>>,
                 Actions::MutateApply<
                     ::dg::PopulateBoundaryMortars<linear_boundary_scheme>>,
                 dg::Actions::ReceiveDataForFluxes<linear_boundary_scheme>,
                 Actions::MutateApply<linear_boundary_scheme>>;

  using solve_linearized_system = tmpl::list<
      typename linear_solver::prepare_solve,
      dg::Actions::InitializeMortars<linear_boundary_scheme, true,
                                     ::Initialization::MergePolicy::Overwrite>,
      ::Actions::WhileNot<LinearSolver::Tags::HasConverged,
                          tmpl::list<typename linear_solver::prepare_step,
                                     build_linear_operator,
                                     typename linear_solver::perform_step>>>;

  using element_array_component = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Phase, Phase::RegisterWithObserver,
              tmpl::list<observers::Actions::RegisterWithObservers<
                             observers::RegisterObservers<
                                 NonlinearSolver::Tags::IterationId,
                                 element_observation_type>>,
                         // We prepare the nonlinear solve here to avoid adding
                         // an extra phase. We can't do it before registration
                         // because it contributes to observers.
                         typename nonlinear_solver::prepare_solve,
                         build_nonlinear_operator,
                         typename nonlinear_solver::update_residual,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::Solve,
              tmpl::list<Actions::RunEventsAndTriggers,
                         NonlinearSolver::Actions::TerminateIfConverged,
                         typename nonlinear_solver::prepare_step,
                         solve_linearized_system,
                         ::Actions::WhileNot<
                             NonlinearSolver::Tags::GlobalizationHasConverged,
                             tmpl::list<typename nonlinear_solver::perform_step,
                                        build_nonlinear_operator,
                                        typename nonlinear_solver::
                                            update_residual>>>>>>;

  // Specify all parallel components that will execute actions at some
  // point.
  using component_list =
      tmpl::flatten<tmpl::list<element_array_component,
                               typename linear_solver::component_list,
                               typename nonlinear_solver::component_list,
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
