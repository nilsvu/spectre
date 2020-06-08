// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/Multigrid/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_mg = TestHelpers::LinearSolver::multigrid;

namespace {

struct KrylovSolver {
  static constexpr Options::String help =
      "Options for the iterative linear solver";
};

struct MultigridSolver {
  static constexpr Options::String help =
      "Options for the iterative linear solver";
};

struct RichardsonSmoother {
  static constexpr Options::String help =
      "Options for the iterative linear solver";
};

struct VcycleDownLabel {};
struct VcycleUpLabel {};
struct PreconditioningLabel {};
struct PrepareLinearSolveLabel {};
struct PrepareLinearSolverStepLabel {};
struct PerformLinearSolverStepLabel {};

struct Metavariables {
  static constexpr const char* const help{
      "Test the Multigrid solver used as a preconditioner for a "
      "Krylov-subspace solver"};

  static constexpr size_t volume_dim = 1;

  using fields_tag = typename helpers_mg::fields_tag;

  using linear_solver =
      LinearSolver::gmres::Gmres<Metavariables, fields_tag, KrylovSolver, true,
                                 LinearSolver::multigrid::Tags::IsFinestLevel>;
  using multigrid = LinearSolver::multigrid::Multigrid<
      Metavariables, typename linear_solver::operand_tag, MultigridSolver,
      typename linear_solver::preconditioner_source_tag>;
  using smoother = LinearSolver::Richardson::Richardson<
      typename multigrid::smooth_fields_tag, RichardsonSmoother,
      typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;

  using Phase = helpers::Phase;

  using initialization_actions =
      tmpl::list<dg::Actions::InitializeDomain<volume_dim>,
                 helpers_mg::InitializeElement,
                 typename linear_solver::initialize_element,
                 typename multigrid::initialize_element,
                 typename smoother::initialize_element,
                 ::Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using register_actions = tmpl::list<typename linear_solver::register_element,
                                      typename multigrid::register_element,
                                      typename smoother::register_element,
                                      Parallel::Actions::TerminatePhase>;

  template <typename Label>
  using smooth_actions = tmpl::list<
      helpers_mg::ComputeOperatorAction<typename smoother::fields_tag>,
      typename smoother::template solve<
          helpers_mg::ComputeOperatorAction<typename smoother::operand_tag>,
          Label>>;

  using solve_actions = tmpl::list<
      helpers_mg::ComputeOperatorAction<typename linear_solver::fields_tag>,
      typename linear_solver::template solve<typename multigrid::template solve<
          smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
          smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>>,
      Parallel::Actions::TerminatePhase>;

  using component_list = tmpl::flatten<tmpl::list<
      typename linear_solver::component_list,
      typename multigrid::component_list, typename smoother::component_list,
      elliptic::DgElementArray<
          Metavariables,
          tmpl::list<Parallel::PhaseActions<Phase, Phase::Initialization,
                                            initialization_actions>,
                     Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                            register_actions>,
                     Parallel::PhaseActions<Phase, Phase::PerformLinearSolve,
                                            solve_actions>>,
          LinearSolver::multigrid::ElementsAllocator<1, MultigridSolver>>,
      observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>,
      helpers::OutputCleaner<Metavariables>>>;
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<linear_solver, multigrid, smoother>>;
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
