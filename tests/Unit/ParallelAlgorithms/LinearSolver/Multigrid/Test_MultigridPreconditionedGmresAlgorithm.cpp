// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/Multigrid/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// \cond
namespace PUP {
class er;
}  // namespace PUP
// \endcond

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_mg = TestHelpers::LinearSolver::multigrid;

namespace {

struct NewtonRaphsonSolver {
  static constexpr Options::String help =
      "Options for the iterative non-linear solver";
};

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

struct Metavariables {
  static constexpr const char* const help{
      "Test the Multigrid solver used as a preconditioner for a "
      "Krylov-subspace solver"};

  static constexpr size_t volume_dim = 1;
  using system =
      TestHelpers::domain::BoundaryConditions::SystemWithoutBoundaryConditions<
          volume_dim>;

  // The test problem is a linear operator, but we add a Newton-Raphson
  // correction scheme to test it in a multigrid context as well.
  using nonlinear_solver = NonlinearSolver::newton_raphson::NewtonRaphson<
      Metavariables, helpers_mg::fields_tag, NewtonRaphsonSolver,
      helpers_mg::sources_tag, LinearSolver::multigrid::Tags::IsFinestGrid>;
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, typename nonlinear_solver::linear_solver_fields_tag,
      KrylovSolver, true, typename nonlinear_solver::linear_solver_source_tag,
      LinearSolver::multigrid::Tags::IsFinestGrid>;
  using multigrid = LinearSolver::multigrid::Multigrid<
      volume_dim, typename linear_solver::operand_tag, MultigridSolver,
      helpers_mg::OperatorIsMassive,
      typename linear_solver::preconditioner_source_tag>;
  using smoother = LinearSolver::Richardson::Richardson<
      typename multigrid::smooth_fields_tag, RichardsonSmoother,
      typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<1>, tmpl::list<domain::creators::Interval>>>;
  };

  static constexpr auto default_phase_order = helpers::default_phase_order;

  using initialization_actions =
      tmpl::list<helpers_mg::InitializeElement,
                 typename nonlinear_solver::initialize_element,
                 typename linear_solver::initialize_element,
                 typename multigrid::initialize_element,
                 typename smoother::initialize_element,
                 Parallel::Actions::TerminatePhase>;

  using register_actions =
      tmpl::list<typename nonlinear_solver::register_element,
                 typename linear_solver::register_element,
                 typename multigrid::register_element,
                 typename smoother::register_element,
                 Parallel::Actions::TerminatePhase>;

  template <typename OperandTag, bool Linear>
  using compute_operator_action = helpers_mg::ComputeOperatorAction<
      OperandTag,
      tmpl::conditional_t<
          Linear,
          db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, OperandTag>,
          db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo,
                             OperandTag>>>;

  template <typename Label>
  using smooth_actions = typename smoother::template solve<
      compute_operator_action<typename smoother::operand_tag, true>, Label>;

  using solve_actions = tmpl::list<
      typename nonlinear_solver::template solve<
          compute_operator_action<typename nonlinear_solver::fields_tag, false>,
          typename linear_solver::template solve<tmpl::list<
              typename multigrid::template solve<
                  compute_operator_action<typename smoother::fields_tag, true>,
                  smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
                  smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>,
              LinearSolver::Actions::make_identity_if_skipped<
                  multigrid, compute_operator_action<
                                 typename linear_solver::operand_tag, true>>>>>,
      Parallel::Actions::TerminatePhase>;

  using test_actions =
      tmpl::list<helpers_mg::TestResult<typename multigrid::options_group>>;

  using component_list = tmpl::flatten<tmpl::list<
      typename nonlinear_solver::component_list,
      typename linear_solver::component_list,
      typename multigrid::component_list, typename smoother::component_list,
      elliptic::DgElementArray<
          Metavariables,
          tmpl::list<
              Parallel::PhaseActions<Parallel::Phase::Initialization,
                                     initialization_actions>,
              Parallel::PhaseActions<Parallel::Phase::Register,
                                     register_actions>,
              Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>,
              Parallel::PhaseActions<Parallel::Phase::Testing, test_actions>>,
          LinearSolver::multigrid::ElementsAllocator<1, MultigridSolver>>,
      observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>,
      helpers::OutputCleaner<Metavariables>>>;
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<nonlinear_solver, linear_solver, multigrid, smoother>>;
  static constexpr bool ignore_unrecognized_command_line_options = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
