// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <cstddef>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainData.hpp"
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
struct InitializeElement {
 private:
  using fields_tag = helpers_distributed::fields_tag;
  using source_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<helpers_distributed::LinearOperator,
                 helpers_distributed::Source,
                 helpers_distributed::ExpectedResult>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& element_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    int array_index = element_index.segments()[0].index();
    auto source = db::item_type<source_tag>{
        gsl::at(get<helpers_distributed::Source>(box), array_index)};
    auto initial_fields =
        make_with_value<db::item_type<fields_tag>>(source, 0.);
    auto operator_applied_to_fields =
        make_with_value<db::item_type<operator_applied_to_fields_tag>>(
            source, std::numeric_limits<double>::signaling_NaN());

    using compute_tags = db::AddComputeTags<>;
    return std::make_tuple(::Initialization::merge_into_databox<
                           InitializeElement,
                           db::AddSimpleTags<fields_tag, source_tag,
                                             operator_applied_to_fields_tag>,
                           compute_tags>(
        std::move(box), std::move(initial_fields), std::move(source),
        std::move(operator_applied_to_fields)));
  }
};

template <typename OptionsGroup>
struct TestResult {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<1>& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    int array_index = element_index.segments()[0].index();
    const auto& has_converged =
        get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             Convergence::Reason::MaxIterations);
    const auto& expected_result =
        gsl::at(get<helpers_distributed::ExpectedResult>(cache), array_index);
    const auto& result = get(get<helpers_distributed::ScalarFieldTag>(box));
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    return {std::move(box), true};
  }
};

struct SubdomainOperator {
  static constexpr size_t volume_dim = 1;
  using SubdomainDataType = LinearSolver::schwarz_detail::SubdomainData<
      volume_dim, db::get_variables_tags_list<helpers_distributed::fields_tag>>;

  using argument_tags =
      tmpl::list<helpers_distributed::LinearOperator, ::Tags::Element<1>>;
  static SubdomainDataType apply(
      const SubdomainDataType& arg,
      const db::item_type<helpers_distributed::LinearOperator>& linear_operator,
      const Element<1>& element) noexcept {
    int array_index = element.id().segment_ids()[0].index();
    const auto& operator_slice = gsl::at(linear_operator, array_index);
    const size_t num_points = operator_slice.columns();
    const DenseMatrix<double, blaze::columnMajor> subdomain_operator =
        blaze::submatrix(operator_slice, array_index * num_points, 0,
                         num_points, num_points);
    SubdomainDataType result{num_points};
    // Apply matrix to central element data
    dgemv_('N', num_points, num_points, 1, subdomain_operator.data(),
           num_points, arg.element_data.data(), 1, 0,
           result.element_data.data(), 1);
    // TODO: Add boundary contributions
    // Parallel::printf("%d operand: %s\n", array_index, arg.element_data);
    // Parallel::printf("%d operator: %s\n", array_index, subdomain_operator);
    // Parallel::printf("%d applied: %s\n", array_index, result.element_data);
    return result;
  }
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;

  using linear_solver =
      LinearSolver::Schwarz<Metavariables, helpers_distributed::fields_tag,
                            SchwarzSolverOptions, SubdomainOperator>;

  using initialization_actions =
      tmpl::list<dg::Actions::InitializeDomain<volume_dim>,
                 InitializeElement<Dim>,
                 typename linear_solver::initialize_element,
                 typename linear_solver::prepare_solve,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using solve_actions =
      tmpl::flatten<tmpl::list<typename linear_solver::prepare_step,
                               LinearSolver::Actions::TerminateIfConverged<
                                   typename linear_solver::options_group>,
                               helpers_distributed::ComputeOperatorAction<
                                   helpers_distributed::fields_tag>,
                               typename linear_solver::perform_step>>;

  enum class Phase { Initialization, Solve, TestResult, Exit };

  using element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<Phase, Phase::Solve, solve_actions>,
          Parallel::PhaseActions<
              Phase, Phase::TestResult,
              tmpl::list<TestResult<typename linear_solver::options_group>>>>>;

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
