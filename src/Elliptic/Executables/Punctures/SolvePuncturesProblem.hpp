// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/ComputeOperatorAction.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/Initialization/FluxLifting.hpp"
#include "Elliptic/Systems/Punctures/Actions/Observe.hpp"
#include "Elliptic/Systems/Punctures/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes2.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication2.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxLifting/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Actions/TerminateIfConverged.hpp"
#include "NumericalAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/GotoAction.hpp"  // IWYU pragma: keep
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticData/Punctures/MultiplePunctures.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct NonlinearSolverStepEnd {};
struct LinearSolverReinitializationEnd {};

struct SkipLinearSolverReinitializationIfNotConverged {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::tuple<db::DataBox<DbTagsList>&&, bool, size_t>(
        std::move(box), false,
        db::get<LinearSolver::Tags::HasConverged>(box)
            ? tmpl::index_of<
                  ActionList,
                  SkipLinearSolverReinitializationIfNotConverged>::value +
                  1
            : tmpl::index_of<
                  ActionList,
                  ::Actions::Label<LinearSolverReinitializationEnd>>::value);
  }
};
struct SkipNonlinearSolverStepIfNotConverged {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::tuple<db::DataBox<DbTagsList>&&, bool, size_t>(
        std::move(box), false,
        db::get<LinearSolver::Tags::HasConverged>(box)
            ? tmpl::index_of<ActionList,
                             SkipNonlinearSolverStepIfNotConverged>::value +
                  1
            : tmpl::index_of<ActionList,
                             ::Actions::Label<NonlinearSolverStepEnd>>::value);
  }
};

}  // namespace

template <size_t Dim>
struct Metavariables {
  static constexpr OptionString help{
      "Find the solution to an Punctures problem in Dim spatial dimensions.\n"
      "Analytic solution: ProductOfSinusoids\n"
      "Linear solver: GMRES\n"
      "Numerical flux: FirstOrderInternalPenaltyFlux"};

  // The system provides all equations specific to the problem.
  using system = Punctures::FirstOrderSystem<Dim>;

  // The analytic solution and corresponding source to solve the Poisson
  // equation for
  using analytic_solution_tag = OptionTags::AnalyticSolution<
      Punctures::InitialGuesses::MultiplePunctures>;
  using initial_guess_tag = analytic_solution_tag;

  using nonlinear_solver = NonlinearSolver::NewtonRaphson<Metavariables>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not positive-definite for the first-order system.
  using linear_solver = LinearSolver::Gmres<Metavariables>;

  using temporal_id =
      Elliptic::Tags::IterationId<NonlinearSolver::Tags::IterationId,
                                  LinearSolver::Tags::IterationId>;

  // Parse numerical flux parameters from the input file to store in the cache.
  using normal_dot_numerical_flux_tag = Elliptic::OptionTags::NumericalFlux<
      Punctures::FirstOrderInternalPenaltyFlux<
          Dim,
          LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
              Punctures::Tags::Field<DataVector>>>,
          LinearSolver::Tags::Operand<
              NonlinearSolver::Tags::Correction<Punctures::Tags::FieldGradient<
                  Dim, Frame::Inertial, DataVector>>>>>;
  using nonlinear_normal_dot_numerical_flux_tag = Elliptic::OptionTags::
      CorrectionNumericalFlux<Punctures::FirstOrderInternalPenaltyFlux<
          Dim, Punctures::Tags::Field<DataVector>,
          Punctures::Tags::FieldGradient<Dim, Frame::Inertial, DataVector>>>;

  using linear_flux_lifting_scheme = dg::FluxLifting::StrongFirstOrder<
      Dim, typename system::variables_tag, typename system::normal_dot_fluxes,
      normal_dot_numerical_flux_tag, LinearSolver::Tags::IterationId>;

  using nonlinear_flux_lifting_scheme = dg::FluxLifting::StrongFirstOrder<
      Dim, typename system::nonlinear_fields_tag,
      typename system::nonlinear_normal_dot_fluxes,
      nonlinear_normal_dot_numerical_flux_tag,
      NonlinearSolver::Tags::IterationId>;

  // Set up the domain creator from the input file.
  using domain_creator_tag = OptionTags::DomainCreator<Dim, Frame::Inertial>;

  // Collect all items to store in the cache.
  using const_global_cache_tag_list =
      tmpl::list<normal_dot_numerical_flux_tag,
                 nonlinear_normal_dot_numerical_flux_tag,
                 analytic_solution_tag>;

  struct ObservationType {};
  using element_observation_type = ObservationType;
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<Punctures::Actions::Observe, linear_solver, nonlinear_solver>>;

  using compute_linear_operator_action = tmpl::list<
      dg::Actions::ComputeFluxes<Tags::InternalDirections<Dim>,
                                 typename system::normal_dot_fluxes>,
      dg::Actions::SendDataForFluxes<linear_flux_lifting_scheme>,
      Elliptic::Actions::ComputeOperatorAction,
      dg::Actions::ComputeFluxes<Tags::BoundaryDirectionsInterior<Dim>,
                                 typename system::normal_dot_fluxes>,
      Elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
          linear_flux_lifting_scheme,
          db::wrap_tags_in<
              LinearSolver::Tags::Operand,
              db::wrap_tags_in<
                  NonlinearSolver::Tags::Correction,
                  typename system::impose_boundary_conditions_on_fields>>>,
      dg::Actions::ReceiveDataForFluxes<linear_flux_lifting_scheme>,
      dg::Actions::ApplyFluxes<linear_flux_lifting_scheme>>;

  using compute_nonlinear_operator_action = tmpl::list<
      dg::Actions::ComputeFluxes<Tags::InternalDirections<Dim>,
                                 typename system::nonlinear_normal_dot_fluxes>,
      dg::Actions::SendDataForFluxes<nonlinear_flux_lifting_scheme>,
      Elliptic::Actions::ComputeNonlinearOperatorAction,
      dg::Actions::ComputeFluxes<Tags::BoundaryDirectionsInterior<Dim>,
                                 typename system::nonlinear_normal_dot_fluxes>,
      Elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
          nonlinear_flux_lifting_scheme,
          typename system::impose_boundary_conditions_on_fields>,
      dg::Actions::ReceiveDataForFluxes<nonlinear_flux_lifting_scheme>,
      dg::Actions::ApplyFluxes<nonlinear_flux_lifting_scheme>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::append<
      tmpl::list<Elliptic::DgElementArray<
          Metavariables,
          tmpl::flatten<tmpl::list<
              SkipLinearSolverReinitializationIfNotConverged,
              Punctures::Actions::Observe,
              NonlinearSolver::Actions::TerminateIfConverged,
              typename linear_solver::tags,
              Elliptic::Initialization::FluxLifting<linear_flux_lifting_scheme>,
              ::Actions::Label<LinearSolverReinitializationEnd>,
              compute_linear_operator_action,
              typename linear_solver::perform_step,
              SkipNonlinearSolverStepIfNotConverged,
              typename nonlinear_solver::perform_step,
              compute_nonlinear_operator_action,
              typename nonlinear_solver::prepare_linear_solve,
              ::Actions::Label<NonlinearSolverStepEnd>>>>>,
      typename linear_solver::component_list,
      typename nonlinear_solver::component_list,
      tmpl::list<observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

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
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
