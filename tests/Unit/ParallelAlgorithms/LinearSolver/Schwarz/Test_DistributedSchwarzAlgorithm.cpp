// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <cstddef>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"

namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct SchwarzSolverOptions {
  static std::string name() noexcept { return "SchwarzSolver"; }
  static constexpr OptionString help =
      "Options for the iterative Schwarz solver";
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;

  using linear_solver =
      LinearSolver::Schwarz<Metavariables, helpers_distributed::fields_tag,
                            SchwarzSolverOptions>;

  using initialization_actions =
      tmpl::list<dg::Actions::InitializeDomain<volume_dim>,
                 typename linear_solver::initialize_element,
                 typename linear_solver::prepare_solve,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using build_linear_operator_actions = tmpl::list<>;

  enum class Phase { Initialization, Solve, TestResult, Exit };

  using element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Initialization,
                                        initialization_actions>,
                 Parallel::PhaseActions<
                     Phase, Phase::Solve,
                     tmpl::flatten<
                         tmpl::list<typename linear_solver::prepare_step,
                                    LinearSolver::Actions::TerminateIfConverged<
                                        typename linear_solver::options_group>,
                                    build_linear_operator_actions,
                                    typename linear_solver::perform_step>>>>>;

  using component_list = tmpl::append<tmpl::list<element_array>,
                                      typename linear_solver::component_list>;

  using observed_reduction_data_tags = tmpl::list<>;

  static constexpr const char* const help{
      "Test the Schwarz linear solver algorithm"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::Solve;
      case Phase::Solve:
        return Phase::TestResult;
      default:
        return Phase::Exit;
    }
  }
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables<1>>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
