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
#include "Parallel/CharmMain.tpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
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
      "Test the Multigrid linear solver algorithm on multiple elements"};

  static constexpr size_t volume_dim = 1;
  using system =
      TestHelpers::domain::BoundaryConditions::SystemWithoutBoundaryConditions<
          volume_dim>;

  // [setup_smoother]
  using multigrid = LinearSolver::multigrid::Multigrid<
      Metavariables, helpers_mg::fields_tag, MultigridSolver,
      helpers_mg::OperatorIsMassive, helpers_mg::sources_tag>;

  using smoother = LinearSolver::Richardson::Richardson<
      typename multigrid::smooth_fields_tag, RichardsonSmoother,
      typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  // [setup_smoother]

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<1>, tmpl::list<domain::creators::Interval>>>;
  };

  static constexpr auto default_phase_order = helpers::default_phase_order;

  using initialization_actions = tmpl::list<
      helpers_mg::InitializeElement, typename multigrid::initialize_element,
      typename smoother::initialize_element, Parallel::Actions::TerminatePhase>;

  using register_actions = tmpl::list<typename multigrid::register_element,
                                      typename smoother::register_element,
                                      Parallel::Actions::TerminatePhase>;

  template <typename OperandTag>
  using compute_operator_action = helpers_mg::ComputeOperatorAction<
      OperandTag,
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, OperandTag>>;

  // [action_list]
  template <typename Label>
  using smooth_actions = typename smoother::template solve<
      compute_operator_action<typename smoother::operand_tag>, Label>;

  using solve_actions =
      tmpl::list<compute_operator_action<typename multigrid::fields_tag>,
                 typename multigrid::template solve<
                     compute_operator_action<typename smoother::fields_tag>,
                     smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
                     smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>,
                 Parallel::Actions::TerminatePhase>;
  // [action_list]

  using test_actions =
      tmpl::list<helpers_mg::TestResult<typename multigrid::options_group>>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<Parallel::Phase::Register, register_actions>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>,
          Parallel::PhaseActions<Parallel::Phase::Testing, test_actions>>,
      LinearSolver::multigrid::ElementsAllocator<1, MultigridSolver>>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = dg_element_array;
  };

  using component_list = tmpl::flatten<tmpl::list<
      typename multigrid::component_list, typename smoother::component_list,
      dg_element_array, observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>,
      helpers::OutputCleaner<Metavariables>>>;
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::list<multigrid, smoother>>;

  static constexpr bool ignore_unrecognized_command_line_options = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm}, {});
}
