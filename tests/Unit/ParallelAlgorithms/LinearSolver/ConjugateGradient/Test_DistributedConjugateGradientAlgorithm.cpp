// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ConjugateGradient.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct ParallelCg {
  static constexpr Options::String help =
      "Options for the iterative linear solver";
};

struct Metavariables {
  static constexpr const char* const help{
      "Test the conjugate gradient linear solver algorithm on multiple "
      "elements"};
  static constexpr size_t volume_dim = 1;
  using system =
      TestHelpers::domain::BoundaryConditions::SystemWithoutBoundaryConditions<
          volume_dim>;

  using linear_solver = LinearSolver::cg::ConjugateGradient<
      Metavariables, typename helpers_distributed::fields_tag, ParallelCg>;
  using preconditioner = void;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<1>, tmpl::list<domain::creators::Interval>>>;
  };

  static constexpr auto default_phase_order = helpers::default_phase_order;
  using component_list = helpers_distributed::component_list<Metavariables>;
  using observed_reduction_data_tags =
      helpers::observed_reduction_data_tags<Metavariables>;
  static constexpr bool ignore_unrecognized_command_line_options = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &domain::creators::register_derived_with_charm,
    &TestHelpers::domain::BoundaryConditions::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
