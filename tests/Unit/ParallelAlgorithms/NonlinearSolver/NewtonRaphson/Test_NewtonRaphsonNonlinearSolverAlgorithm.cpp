// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <vector>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/NonlinearSolver/NonlinearSolverAlgorithmTestHelpers.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = NonlinearSolverAlgorithmTestHelpers;

namespace {

struct LinearSolverGroup {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr Options::String help = "Options for the linear solver";
};

struct NonlinearSolverGroup {
  static std::string name() noexcept { return "NewtonRaphson"; }
  static constexpr Options::String help = "Options for the nonlinear solver";
};

struct Metavariables {
  static constexpr const char* const help{
      "Test the Newton-Raphson nonlinear solver algorithm"};

  using nonlinear_solver = NonlinearSolver::newton_raphson::NewtonRaphson<
      Metavariables, helpers::fields_tag, NonlinearSolverGroup>;
  using linear_solver = LinearSolver::Richardson::Richardson<
      typename nonlinear_solver::linear_solver_fields_tag, LinearSolverGroup,
      typename nonlinear_solver::linear_solver_source_tag>;

  enum class Phase {
    Initialization,
    RegisterWithObserver,
    Solve,
    TestResult,
    CleanOutput,
    Exit
  };

  using component_list =
      tmpl::append<tmpl::list<helpers::ElementArray<Metavariables>,
                              observers::Observer<Metavariables>,
                              observers::ObserverWriter<Metavariables>,
                              helpers::OutputCleaner<Metavariables>>,
                   typename nonlinear_solver::component_list,
                   typename linear_solver::component_list>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<nonlinear_solver, linear_solver>>;

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
        return Phase::TestResult;
      case Phase::TestResult:
        return Phase::CleanOutput;
      default:
        return Phase::Exit;
    }
  }

  static constexpr bool ignore_unrecognized_command_line_options = false;
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
