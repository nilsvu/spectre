// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/ComputeOperatorAction.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeBackgroundFields.hpp"
#include "Elliptic/Actions/InitializeIterationIds.hpp"
#include "Elliptic/Actions/InitializeNonlinearSystem.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeFluxes.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/Systems/Xcts/Actions/LapseAtOrigin.hpp"
#include "Elliptic/Systems/Xcts/Actions/Observe.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication2.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeInterfaces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/InternalPenalty.hpp"
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
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Globalization/LineSearch/LineSearch.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/NeutronStarBinary.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct CorrectionNumericalFluxGroup {
  static std::string name() noexcept { return "CorrectionNumFlux"; }
  static constexpr OptionString help =
      "The numerical flux scheme for the correction quantity";
};

template <typename NumericalFluxType>
struct CorrectionNumericalFlux {
  static std::string name() noexcept {
    return option_name<NumericalFluxType>();
  }
  static constexpr OptionString help =
      "Options for the correction numerical flux";
  using type = NumericalFluxType;
  using group = CorrectionNumericalFluxGroup;
  using container_tag =
      ::Tags::NormalDotNumericalFluxComputer<NumericalFluxType>;
};

struct NonlinearSolverStepStart {};
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
struct RepeatNonlinearGlobalizationIfNotConverged {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>&
                    /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/, const ParallelComponent* const
                    /*meta*/) noexcept {
    return std::tuple<db::DataBox<DbTagsList>&&, bool, size_t>(
        std::move(box), false,
        db::get<NonlinearSolver::Tags::GlobalizationHasConverged>(box)
            ? tmpl::index_of<
                  ActionList,
                  RepeatNonlinearGlobalizationIfNotConverged>::value +
                  1
            : tmpl::index_of<ActionList, ::Actions::Label<
                                             NonlinearSolverStepStart>>::value);
  }
};

}  // namespace

template <size_t Dim>
struct Metavariables {
  static constexpr OptionString help{
      "Find the solution to an XCTS problem in Dim spatial dimensions.\n"
      "Analytic solution: ProductOfSinusoids\n"
      "Linear solver: GMRES\n"
      "Numerical flux: FirstOrderInternalPenaltyFlux"};

  // The system provides all equations specific to the problem.
  using system = Xcts::FirstOrderHamiltonianAndLapseSystem<Dim>;
  using nonlinear_fields_tag = typename system::fields_tag;

  using linearized_system = typename system::linearized_system;
  using linear_fields_tag = typename linearized_system::fields_tag;

  // The analytic solution and corresponding source to solve the XCTS
  // equation for
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<Xcts::AnalyticData::NeutronStarBinary>;
  using initial_guess_tag = analytic_solution_tag;

  using nonlinear_solver = NonlinearSolver::NewtonRaphson<
      Metavariables, nonlinear_fields_tag,
      NonlinearSolver::Globalization::LineSearch>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not positive-definite for the first-order system.
  using linear_solver = LinearSolver::Gmres<Metavariables, linear_fields_tag>;
  using linear_operand_tag = typename linear_solver::operand_tag;

  // Specify the DG boundary scheme. We use the strong first-order scheme here
  // that only requires us to compute normals dotted into the first-order
  // fluxes.
  using nonlinear_numerical_flux =
      dg::NumericalFluxes::FirstOrderInternalPenalty<
          Dim, typename system::primal_variables,
          typename system::auxiliary_variables>;
  using nonlinear_boundary_scheme = dg::BoundarySchemes::StrongFirstOrder<
      Dim, nonlinear_fields_tag,
      Tags::NormalDotNumericalFluxComputer<nonlinear_numerical_flux>,
      NonlinearSolver::Tags::TemporalId>;
  using linear_numerical_flux = dg::NumericalFluxes::FirstOrderInternalPenalty<
      Dim, typename linearized_system::primal_variables,
      typename linearized_system::auxiliary_variables>;
  using linear_boundary_scheme = dg::BoundarySchemes::StrongFirstOrder<
      Dim, linear_operand_tag,
      Tags::NormalDotNumericalFluxComputer<linear_numerical_flux>,
      LinearSolver::Tags::IterationId>;

  // Set up the domain creator from the input file.
  using domain_creator_tag = OptionTags::DomainCreator<Dim, Frame::Inertial>;

  // Collect all items to store in the cache.
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::NumericalFlux<linear_numerical_flux>,
                 CorrectionNumericalFlux<nonlinear_numerical_flux>,
                 analytic_solution_tag>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<Xcts::Actions::Observe, linear_solver, nonlinear_solver>>;

  // Specify all global synchronization points.
  enum class Phase {
    Initialization,
    RegisterWithObserver,
    Observe,
    Solve,
    Exit
  };

  // Construct the DgElementArray parallel component
  using apply_nonlinear_operator = tmpl::list<
      Xcts::Actions::UpdateLapseAtOrigin,
      dg::Actions::SendDataForFluxes<nonlinear_boundary_scheme>,
      elliptic::Actions::ComputeOperatorAction<
          Dim, NonlinearSolver::Tags::OperatorAppliedTo, nonlinear_fields_tag>,
      Actions::MutateApply<
          ::dg::PopulateBoundaryMortars<nonlinear_boundary_scheme>>,
      dg::Actions::ReceiveDataForFluxes<nonlinear_boundary_scheme>,
      Actions::MutateApply<nonlinear_boundary_scheme>>;

  using apply_linear_operator = tmpl::list<
      dg::Actions::SendDataForFluxes<linear_boundary_scheme>,
      elliptic::Actions::ComputeOperatorAction<
          Dim, LinearSolver::Tags::OperatorAppliedTo, linear_operand_tag>,
      Actions::MutateApply<
          ::dg::PopulateBoundaryMortars<linear_boundary_scheme>>,
      dg::Actions::ReceiveDataForFluxes<linear_boundary_scheme>,
      Actions::MutateApply<linear_boundary_scheme>>;

  using initialization_actions = tmpl::flatten<tmpl::list<
      dg::Actions::InitializeDomain<Dim>,
      elliptic::Actions::InitializeAnalyticSolution,
      //   elliptic::Actions::InitializeBackgroundFields,
      Xcts::Actions::InitializeLapseAtOrigin,
      elliptic::Actions::InitializeNonlinearSystem,
      dg::Actions::InitializeInterfaces<
          system, dg::Initialization::slice_tags_to_face<>,
          dg::Initialization::slice_tags_to_exterior<>>,
      elliptic::dg::Actions::InitializeFluxes<nonlinear_boundary_scheme,
                                              system>,
      elliptic::dg::Actions::InitializeFluxes<linear_boundary_scheme,
                                              linearized_system, false>,
      elliptic::Actions::InitializeIterationIds<
          NonlinearSolver::Tags::IterationId, LinearSolver::Tags::IterationId>,
      typename nonlinear_solver::globalization_strategy::initialize_element,
      dg::Actions::InitializeMortars<linear_boundary_scheme, true>,
      dg::Actions::InitializeMortars<nonlinear_boundary_scheme, true>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          nonlinear_fields_tag, nonlinear_boundary_scheme>,
      apply_nonlinear_operator,
      dg::Actions::InitializeMortars<nonlinear_boundary_scheme, true,
                                     ::Initialization::MergePolicy::Overwrite>,
      typename nonlinear_solver::initialize_element,
      typename linear_solver::initialize_element,
      Parallel::Actions::TerminatePhase>>;

  using element_array_component = elliptic::DgElementArray<
      Metavariables,
      Parallel::ForwardAllOptionsToDataBox<
          Initialization::option_tags<initialization_actions>>,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Phase, Phase::RegisterWithObserver,
              tmpl::list<observers::Actions::RegisterWithObservers<
                             Xcts::Actions::Observe>,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<Phase, Phase::Observe,
                                 tmpl::list<Xcts::Actions::Observe,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::Solve,
              tmpl::flatten<tmpl::list<
                  SkipLinearSolverReinitializationIfNotConverged,
                  Xcts::Actions::Observe,
                  NonlinearSolver::Actions::TerminateIfConverged,
                  typename linear_solver::reinitialize_element,
                  dg::Actions::InitializeMortars<
                      linear_boundary_scheme, true,
                      ::Initialization::MergePolicy::Overwrite>,
                  Actions::Label<LinearSolverReinitializationEnd>,
                  apply_linear_operator, typename linear_solver::perform_step,
                  SkipNonlinearSolverStepIfNotConverged,
                  Actions::Label<NonlinearSolverStepStart>,
                  typename nonlinear_solver::perform_step,
                  apply_nonlinear_operator,
                  typename nonlinear_solver::apply_globalization,
                  RepeatNonlinearGlobalizationIfNotConverged,
                  typename nonlinear_solver::globalization_strategy::
                      reinitialize_element,
                  Actions::Label<NonlinearSolverStepEnd>>>>>>;

  // Specify all parallel components that will execute actions at some point.
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
        return Phase::Observe;
      case Phase::Observe:
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
