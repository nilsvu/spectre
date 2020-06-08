// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct MultigridSolver {
  static constexpr OptionString help =
      "Options for the iterative linear solver";
};

struct RichardsonSmoother {
  static constexpr OptionString help =
      "Options for the iterative linear solver";
};

struct VcycleDownLabel {};
struct VcycleUpLabel {};

struct Metavariables {
  static constexpr const char* const help{
      "Test the Multigrid linear solver algorithm on multiple elements"};

  static constexpr size_t volume_dim = 1;

  static constexpr bool massive_operator = true;

  using fields_tag = typename helpers_distributed::fields_tag;

  using linear_solver =
      LinearSolver::multigrid::Multigrid<Metavariables, fields_tag,
                                         MultigridSolver>;

  using smoother = LinearSolver::Richardson::Richardson<
      typename linear_solver::smooth_fields_tag, RichardsonSmoother,
      typename linear_solver::smooth_source_tag>;

  using Phase = helpers::Phase;

  using initialization_actions =
      tmpl::list<dg::Actions::InitializeDomain<volume_dim>,
                 helpers_distributed::InitializeElement,
                 typename linear_solver::initialize_element,
                 typename smoother::initialize_element,
                 helpers_distributed::ComputeOperatorAction<fields_tag>,
                 ::Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using register_actions = tmpl::list<typename linear_solver::register_element,
                                      typename smoother::register_element,
                                      typename linear_solver::prepare_solve,
                                      Parallel::Actions::TerminatePhase>;

  template <typename LabelTag>
  using smooth_actions = tmpl::list<
      helpers_distributed::ComputeOperatorAction<typename smoother::fields_tag>,
      typename smoother::prepare_solve,
      ::Actions::RepeatUntil<
          LinearSolver::Tags::HasConverged<typename smoother::options_group>,
          tmpl::list<typename smoother::prepare_step,
                     helpers_distributed::ComputeOperatorAction<
                         typename smoother::operand_tag>,
                     typename smoother::perform_step>,
          LabelTag>>;

  using solve_actions =
      tmpl::list<LinearSolver::Actions::TerminateIfConverged<
                     typename linear_solver::options_group>,
                 typename linear_solver::prepare_step_down,
                 smooth_actions<VcycleDownLabel>,
                 helpers_distributed::ComputeOperatorAction<
                     typename linear_solver::operand_tag>,
                 typename linear_solver::perform_step_down,
                 typename linear_solver::prepare_step_up,
                 smooth_actions<VcycleUpLabel>,
                 helpers_distributed::ComputeOperatorAction<
                     typename linear_solver::operand_tag>,
                 typename linear_solver::perform_step_up>;

  using component_list = tmpl::flatten<tmpl::list<
      typename linear_solver::component_list, typename smoother::component_list,
      elliptic::DgElementArray<
          Metavariables,
          tmpl::list<Parallel::PhaseActions<Phase, Phase::Initialization,
                                            initialization_actions>,
                     Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                            register_actions>,
                     Parallel::PhaseActions<Phase, Phase::PerformLinearSolve,
                                            solve_actions>>,
          LinearSolver::multigrid::ElementsAllocator<1>>,
      observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>,
      helpers::OutputCleaner<Metavariables>>>;
  using observed_reduction_data_tags =
      tmpl::append<typename linear_solver::observed_reduction_data_tags,
                   typename smoother::observed_reduction_data_tags>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  static constexpr auto determine_next_phase =
      helpers::determine_next_phase<Metavariables>;
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
