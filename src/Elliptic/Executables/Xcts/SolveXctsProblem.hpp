// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/ApplyLinearOperatorToInitialFields.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeBackgroundFields.hpp"
// #include "Elliptic/Actions/InitializeBoundaryConditions.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/Actions/InitializeLinearOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
// #include
// "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFirstOrderOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
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
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveVolumeIntegrals.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Actions/RestrictFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/ResetSubdomainPreconditioner.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainPreconditioners/ExplicitInverse.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace SolveXctsProblem {
namespace OptionTags {
struct NonlinearSolverGroup {
  static std::string name() noexcept { return "NonlinearSolver"; }
  static constexpr Options::String help = "The iterative nonlinear solver";
};
struct NewtonRaphsonGroup {
  static std::string name() noexcept { return "NewtonRaphson"; }
  static constexpr Options::String help =
      "Options for the Newton-Raphson nonlinear solver";
  using group = NonlinearSolverGroup;
};
struct LinearSolverGroup {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr Options::String help =
      "The iterative Krylov-subspace linear solver";
  using group = NonlinearSolverGroup;
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
  using argument_tags = tmpl::list<Tags...>;
  using type = tuples::TaggedTuple<Tags...>;
  static type function(
      const db::const_item_type<Tags>&... components) noexcept {
    return {components...};
  }
  template <typename Tag>
  using step_prefix = LinearSolver::Tags::OperatorAppliedTo<Tag>;
};
template <typename NumericalFluxType, typename OptionTag>
struct NonlinNumericalFluxTag : db::SimpleTag {
  using type = NumericalFluxType;
  using option_tags = tmpl::list<OptionTag>;

  static constexpr bool pass_metavariables = false;
  template <typename OptionCreatedNumFlux>
  static NumericalFluxType create_from_options(
      const OptionCreatedNumFlux& numerical_flux) noexcept {
    return numerical_flux;
  }
};
}  // namespace SolveXctsProblem

/// \cond
template <typename System, typename Background, typename BoundaryConditions,
          typename InitialGuess>
struct Metavariables {
  using system = System;
  using system_fields = typename system::fields_tag::tags_list;
  using linearized_system = typename System::linearized_system;
  static constexpr size_t volume_dim = system::volume_dim;
  using background = Background;
  using boundary_conditions = BoundaryConditions;
  using initial_guess = InitialGuess;

  static constexpr bool massive_operator = true;

  static constexpr Options::String help{
      "Find the solution to a nonlinear XCTS problem."};

  using nonlinear_fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;
  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename linearized_system::fluxes>;

  static constexpr bool has_analytic_solution =
      tt::conforms_to_v<background, elliptic::protocols::AnalyticSolution>;

  using background_tag =
      tmpl::conditional_t<has_analytic_solution,
                          ::Tags::AnalyticSolution<background>,
                          elliptic::Tags::Background<background>>;
  using boundary_conditions_tag =
      ::Tags::BoundaryCondition<boundary_conditions>;
  using linearized_boundary_conditions_tag =
      ::Tags::BoundaryCondition<typename boundary_conditions::linearization>;
  using initial_guess_tag = elliptic::Tags::InitialGuess<initial_guess>;

  using nonlinear_solver = NonlinearSolver::newton_raphson::NewtonRaphson<
      Metavariables, typename system::fields_tag,
      SolveXctsProblem::OptionTags::NewtonRaphsonGroup,
      LinearSolver::multigrid::Tags::IsFinestLevel>;
  using nonlinear_solver_iteration_id =
      LinearSolver::Tags::IterationId<typename nonlinear_solver::options_group>;
  using nonlinear_operand_tag = typename nonlinear_solver::operand_tag;
  using nonlinear_primal_variables = typename system::primal_fields;
  using nonlinear_auxiliary_variables = typename system::auxiliary_fields;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not positive-definite for the first-order system.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, typename nonlinear_solver::linear_solver_fields_tag,
      SolveXctsProblem::OptionTags::GmresGroup, true,
      typename nonlinear_solver::linear_solver_source_tag,
      LinearSolver::multigrid::Tags::IsFinestLevel>;
  using linear_solver_iteration_id =
      LinearSolver::Tags::IterationId<typename linear_solver::options_group>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using linear_operand_tag = typename linear_solver::operand_tag;
  using primal_variables = db::wrap_tags_in<
      LinearSolver::Tags::Preconditioned,
      db::wrap_tags_in<LinearSolver::Tags::Operand,
                       typename linearized_system::primal_fields>>;
  using auxiliary_variables = db::wrap_tags_in<
      LinearSolver::Tags::Preconditioned,
      db::wrap_tags_in<LinearSolver::Tags::Operand,
                       typename linearized_system::auxiliary_fields>>;
  static_assert(
      std::is_same_v<typename linear_operand_tag::tags_list,
                     tmpl::append<primal_variables, auxiliary_variables>>,
      "The primal and auxiliary variables must compose the linear operand (in "
      "the correct order)");

  using multigrid = LinearSolver::multigrid::Multigrid<
      Metavariables, linear_operand_tag,
      SolveXctsProblem::OptionTags::MultigridGroup,
      typename linear_solver::preconditioner_source_tag>;
  using preconditioner_iteration_id =
      LinearSolver::Tags::IterationId<typename multigrid::options_group>;

  // Parse numerical flux parameters from the input file to store in the cache.
  using linear_normal_dot_numerical_flux = Tags::NumericalFlux<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, primal_variables,
          auxiliary_variables>>;
  using normal_dot_numerical_flux = SolveXctsProblem::NonlinNumericalFluxTag<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, nonlinear_fluxes_computer_tag, nonlinear_primal_variables,
          nonlinear_auxiliary_variables>,
      ::OptionTags::NumericalFlux<
          typename linear_normal_dot_numerical_flux::type>>;

  // Disabling the boundary conditions in the subdomain operator means that
  // homogeneous boundary conditions are used instead. This is more robust
  // against changes in the nonlinear background fields, so we may not have to
  // reset the subdomain preconditioner between nonlinear steps, thus speeding
  // up the run at the cost of more linear solver iterations.
  static constexpr bool disable_subdomain_boundary_conditions = true;
  using smoother_subdomain_operator = elliptic::dg::SubdomainOperator<
      volume_dim, primal_variables, auxiliary_variables, fluxes_computer_tag,
      tmpl::list<>, typename linearized_system::sources, tmpl::list<>,
      linear_normal_dot_numerical_flux, linearized_boundary_conditions_tag,
      SolveXctsProblem::OptionTags::SchwarzGroup, tmpl::list<>,
      massive_operator, disable_subdomain_boundary_conditions>;
  using communicated_overlap_fields = tmpl::flatten<tmpl::list<
      typename system::primal_fields,
      db::wrap_tags_in<::Tags::Flux, typename system::primal_fields,
                       tmpl::size_t<volume_dim>, Frame::Inertial>,
      tmpl::conditional_t<
          std::is_same_v<Xcts::BoundaryConditions::ApparentHorizon<
                             Xcts::Geometry::Euclidean>,
                         boundary_conditions> or
              std::is_same_v<Xcts::BoundaryConditions::ApparentHorizon<
                                 Xcts::Geometry::NonEuclidean>,
                             boundary_conditions> or
              std::is_same_v<Xcts::BoundaryConditions::Binary<
                                 Xcts::BoundaryConditions::ApparentHorizon<
                                     Xcts::Geometry::Euclidean>>,
                             boundary_conditions> or
              std::is_same_v<Xcts::BoundaryConditions::Binary<
                                 Xcts::BoundaryConditions::ApparentHorizon<
                                     Xcts::Geometry::NonEuclidean>>,
                             boundary_conditions>,
          tmpl::list<domain::Tags::Interface<
                         domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                         Xcts::Tags::ConformalFactor<DataVector>>,
                     domain::Tags::Interface<
                         domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                         Xcts::Tags::LapseTimesConformalFactor<DataVector>>,
                     domain::Tags::Interface<
                         domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                         ::Tags::NormalDotFlux<Xcts::Tags::ShiftExcess<
                             DataVector, volume_dim, Frame::Inertial>>>>,
          tmpl::list<>>>>;
  using smoother_subdomain_preconditioner =
      LinearSolver::Schwarz::subdomain_preconditioners::ExplicitInverse<
          volume_dim>;
  using smoother = LinearSolver::Schwarz::Schwarz<
      Metavariables, typename multigrid::smooth_fields_tag,
      SolveXctsProblem::OptionTags::SchwarzGroup, smoother_subdomain_operator,
      smoother_subdomain_preconditioner, typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  using smoother_iteration_id =
      LinearSolver::Tags::IterationId<typename smoother::options_group>;

  using combined_iteration_id =
      SolveXctsProblem::CombinedIterationId<linear_solver_iteration_id,
                                            preconditioner_iteration_id,
                                            smoother_iteration_id>;

  // Specify the DG boundary scheme. We use the strong first-order scheme here
  // that only requires us to compute normals dotted into the first-order
  // fluxes.
  using nonlinear_boundary_scheme = dg::FirstOrderScheme::FirstOrderScheme<
      volume_dim, nonlinear_operand_tag, normal_dot_numerical_flux,
      nonlinear_solver_iteration_id, massive_operator,
      NonlinearSolver::Tags::OperatorAppliedTo>;
  using boundary_scheme = dg::FirstOrderScheme::FirstOrderScheme<
      volume_dim, linear_operand_tag, linear_normal_dot_numerical_flux,
      combined_iteration_id, massive_operator,
      LinearSolver::Tags::OperatorAppliedTo>;

  // Collect events and triggers
  // (public for use by the Charm++ registration code)
  using observe_fields = tmpl::flatten<
      tmpl::list<system_fields, typename system::background_fields,
                 db::wrap_tags_in<NonlinearSolver::Tags::Residual,
                                  typename nonlinear_operand_tag::tags_list>>>;
  using analytic_solution_fields =
      tmpl::conditional_t<has_analytic_solution, system_fields, tmpl::list<>>;
  using events = tmpl::flatten<tmpl::list<
      dg::Events::Registrars::ObserveFields<
          volume_dim, nonlinear_solver_iteration_id, observe_fields,
          analytic_solution_fields,
          LinearSolver::multigrid::Tags::MultigridLevel>,
      tmpl::conditional_t<
          has_analytic_solution,
          dg::Events::Registrars::ObserveErrorNorms<
              nonlinear_solver_iteration_id, analytic_solution_fields,
              LinearSolver::multigrid::Tags::MultigridLevel>,
          tmpl::list<>>>>;
  using triggers = tmpl::list<elliptic::Triggers::Registrars::EveryNIterations<
      nonlinear_solver_iteration_id,
      LinearSolver::multigrid::Tags::IsFinestLevel>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags = tmpl::flatten<
      tmpl::list<background_tag, initial_guess_tag, boundary_conditions_tag,
                 linearized_boundary_conditions_tag, fluxes_computer_tag,
                 nonlinear_fluxes_computer_tag, normal_dot_numerical_flux,
                 linear_normal_dot_numerical_flux,
                 Tags::EventsAndTriggers<events, triggers>>>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<
          tmpl::list<typename Event<events>::creatable_classes,
                     nonlinear_solver, linear_solver, multigrid, smoother>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<volume_dim>,
      elliptic::Actions::InitializeFixedSources,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              ::Tags::Variables<typename system::background_fields>>,
          dg::Initialization::slice_tags_to_exterior<
              ::Tags::Variables<typename system::background_fields>>,
          dg::Initialization::face_compute_tags<
              domain::Tags::BoundaryCoordinates<volume_dim>>,
          dg::Initialization::exterior_compute_tags<>, false, false>,
      elliptic::Actions::InitializeFields,
      typename nonlinear_solver::initialize_element,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename smoother::initialize_element,
      elliptic::dg::Actions::InitializeSubdomain<
          volume_dim, typename smoother::options_group,
          linearized_boundary_conditions_tag,
          typename system::background_fields, communicated_overlap_fields,
          primal_variables, typename system::inv_metric_tag>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<combined_iteration_id>>,
      tmpl::conditional_t<has_analytic_solution,
                          elliptic::Actions::InitializeAnalyticSolution<
                              background_tag, analytic_solution_fields>,
                          tmpl::list<>>,
      //   elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
      //       Metavariables>,
      dg::Actions::InitializeMortars<nonlinear_boundary_scheme>,
      dg::Actions::InitializeMortars<boundary_scheme>,
      elliptic::dg::Actions::InitializeFirstOrderOperator<
          volume_dim, typename system::fluxes, typename system::sources,
          nonlinear_operand_tag, nonlinear_primal_variables,
          nonlinear_auxiliary_variables>,
      elliptic::dg::Actions::InitializeFirstOrderOperator<
          volume_dim, typename linearized_system::fluxes,
          typename linearized_system::sources, linear_operand_tag,
          primal_variables, auxiliary_variables>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using build_nonlinear_operator_actions =
      tmpl::list<dg::Actions::CollectDataForFluxes<
                     nonlinear_boundary_scheme,
                     domain::Tags::InternalDirections<volume_dim>>,
                 dg::Actions::SendDataForFluxes<nonlinear_boundary_scheme>,
                 Actions::MutateApply<elliptic::FirstOrderOperator<
                     volume_dim, NonlinearSolver::Tags::OperatorAppliedTo,
                     nonlinear_operand_tag, massive_operator>>,
                 elliptic::dg::Actions::ImposeBoundaryConditions<
                     boundary_conditions_tag, nonlinear_operand_tag,
                     nonlinear_primal_variables, nonlinear_auxiliary_variables,
                     nonlinear_fluxes_computer_tag>,
                 dg::Actions::CollectDataForFluxes<
                     nonlinear_boundary_scheme,
                     domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
                 dg::Actions::ReceiveDataForFluxes<nonlinear_boundary_scheme>,
                 Actions::MutateApply<nonlinear_boundary_scheme>>;

  using build_linear_operator_actions = tmpl::list<
      dg::Actions::CollectDataForFluxes<
          boundary_scheme, domain::Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<boundary_scheme>,
      Actions::MutateApply<elliptic::FirstOrderOperator<
          volume_dim, LinearSolver::Tags::OperatorAppliedTo, linear_operand_tag,
          massive_operator>>,
      elliptic::dg::Actions::ImposeBoundaryConditions<
          linearized_boundary_conditions_tag, linear_operand_tag,
          primal_variables, auxiliary_variables, fluxes_computer_tag>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      Actions::MutateApply<boundary_scheme>>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename nonlinear_solver::register_element,
                 typename linear_solver::register_element,
                 typename multigrid::register_element,
                 typename smoother::register_element,
                 Parallel::Actions::TerminatePhase>;

  template <typename Label>
  using smooth_actions = tmpl::list<
      build_linear_operator_actions,
      typename smoother::template solve<build_linear_operator_actions, Label>>;

  using solve_actions = tmpl::list<
      build_nonlinear_operator_actions,
      typename nonlinear_solver::template solve<
          build_nonlinear_operator_actions,
          tmpl::list<
              Actions::RunEventsAndTriggers,
              LinearSolver::multigrid::Actions::SendFieldsToCoarserGrid<
                  typename system::fields_tag,
                  typename multigrid::options_group>,
              LinearSolver::multigrid::Actions::ReceiveFieldsFromFinerGrid<
                  volume_dim, typename system::fields_tag,
                  typename multigrid::options_group>,
              LinearSolver::Schwarz::Actions::SendOverlapFields<
                  communicated_overlap_fields,
                  typename smoother::options_group>,
              LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
                  volume_dim, communicated_overlap_fields,
                  typename smoother::options_group>,
              // TODO: don't reset subdomain prec in every step, because it
              // might still be good enough.
              // This breaks when boundary conditions depend on the nonlinear
              // fields. Perhaps we can compute some estimator to determine when
              // the subdomain preconditioner needs to be reset.
              // It may even be faster to always use homogeneous boundary
              // conditions on the subdomains and never reset the subdomain
              // preconditioner.
              //   LinearSolver::Schwarz::Actions::ResetSubdomainPreconditioner<
              //       typename smoother::options_group>,
              typename linear_solver::template solve<
                  typename multigrid::template solve<
                      smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
                      smooth_actions<
                          LinearSolver::multigrid::VcycleUpLabel>>>>>,
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
      dg_element_array, typename nonlinear_solver::component_list,
      typename linear_solver::component_list,
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
