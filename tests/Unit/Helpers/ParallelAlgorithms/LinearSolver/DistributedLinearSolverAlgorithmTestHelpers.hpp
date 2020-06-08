// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementId.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;

/// Functionality to test parallel linear solvers on multiple elements
namespace DistributedLinearSolverAlgorithmTestHelpers {

namespace OptionTags {
// This option expects a list of N matrices that each have N*M rows and M
// columns, where N is the number of elements and M is a nonzero integer.
// Therefore, this option specifies a (N*M,N*M) matrix that has its columns
// split over all elements. In a context where the linear operator represents a
// DG discretization, M is the number of collocation points per element.
struct LinearOperator {
  static constexpr OptionString help = "The linear operator A to invert.";
  using type = std::vector<std::vector<DenseMatrix<double, blaze::columnMajor>>>;
};
// Both of the following options expect a list of N vectors that have a size of
// M each, so that they constitute a vector of total size N*M (see above).
struct Source {
  static constexpr OptionString help = "The source b in the equation Ax=b.";
  using type = std::vector<DenseVector<double>>;
};
struct ExpectedResult {
  static constexpr OptionString help = "The solution x in the equation Ax=b";
  using type = std::vector<DenseVector<double>>;
};
}  // namespace OptionTags

struct LinearOperator : db::SimpleTag {
  using type = std::vector<std::vector<DenseMatrix<double, blaze::columnMajor>>>;
  using option_tags = tmpl::list<OptionTags::LinearOperator>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& linear_operator) noexcept {
    return linear_operator;
  }
};

struct Source : db::SimpleTag {
  using type = std::vector<DenseVector<double>>;
  using option_tags = tmpl::list<OptionTags::Source>;

  static constexpr bool pass_metavariables = false;
  static std::vector<DenseVector<double>> create_from_options(
      const std::vector<DenseVector<double>>& source) noexcept {
    return source;
  }
};

struct ExpectedResult : db::SimpleTag {
  using type = std::vector<DenseVector<double>>;
  using option_tags = tmpl::list<OptionTags::ExpectedResult>;

  static constexpr bool pass_metavariables = false;
  static std::vector<DenseVector<double>> create_from_options(
      const std::vector<DenseVector<double>>& expected_result) noexcept {
    return expected_result;
  }
};

// The vector `x` we want to solve for
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarField"; }
};

using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
using sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
using operator_applied_to_fields_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

// We are working on a one-dimension domain where each element corresponds to an
// entry of the vectors provided in the input file. This function translates the
// element to an index into these vectors. We assume the domain is composed of a
// single block.
size_t get_index(const ElementId<1>& element_id) noexcept {
  return element_id.segment_ids()[0].index();
}

// In the following `ComputeOperatorAction` and `CollectOperatorAction` actions
// we compute A(p)=sum_elements(A_element(p_element)) in a global reduction and
// then broadcast the global A(p) back to the elements so that they can extract
// their A_element(p). This is horribly inefficient parallelism but allows us to
// just provide a global matrix A (represented by the `LinearOperator` tag) in
// an input file.

// Forward declare to keep these actions in the order they are used
template <typename OperandTag>
struct CollectOperatorAction;

template <typename OperandTag>
struct ComputeOperatorAction {
  using const_global_cache_tags = tmpl::list<LinearOperator>;
  using local_operator_applied_to_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, OperandTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ElementId<1>& element_id,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t mg_lev =
        db::get<LinearSolver::multigrid::Tags::MultigridLevel>(box);
    const size_t element_index = get_index(element_id);
    const auto& operator_matrices = get<LinearOperator>(box)[mg_lev];
    const auto number_of_elements = operator_matrices.size();
    const auto& linear_operator = gsl::at(operator_matrices, element_index);
    const auto number_of_grid_points = linear_operator.columns();
    const auto& operand = get<OperandTag>(box);

    db::item_type<OperandTag> operator_applied_to_operand{
        number_of_grid_points * number_of_elements};
    // Parallel::printf("%s operand: %s\n", element_id,
    //                  operand);
    dgemv_('N', linear_operator.rows(), linear_operator.columns(), 1,
           linear_operator.data(), linear_operator.spacing(), operand.data(), 1,
           0, operator_applied_to_operand.data(), 1);
    // Parallel::printf("%s operator_applied_to_operand: %s\n", element_id,
    //                  operator_applied_to_operand);

    // Parallel::contribute_to_reduction<CollectOperatorAction<OperandTag>>(
    //     Parallel::ReductionData<
    //         Parallel::ReductionDatum<db::item_type<OperandTag>,
    //         funcl::Plus<>>>{ operator_applied_to_operand},
    //     Parallel::get_parallel_component<ParallelComponent>(cache)[element_id],
    //     Parallel::get_parallel_component<ParallelComponent>(cache));
    // Parallel::printf("%s contribute to section reduction\n", element_id);
    db::mutate<LinearSolver::multigrid::Tags::ArraySectionBase<
        LinearSolver::multigrid::Tags::MultigridLevel>>(
        make_not_null(&box), [&operator_applied_to_operand, &element_id,
                              &mg_lev](const auto section) {
          Parallel::contribute_to_section_reduction<
              ParallelComponent, CollectOperatorAction<OperandTag>>(
              Parallel::ReductionData<
                  Parallel::ReductionDatum<db::item_type<OperandTag>,
                                           funcl::Plus<>>,
                  Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>>{
                  operator_applied_to_operand, mg_lev},
              (*section)[element_id], *section, *section, mg_lev);
        });

    // Terminate algorithm for now. The reduction will be broadcast to the
    // next action which is responsible for restarting the algorithm.
    return {std::move(box), true};
  }
};

template <typename OperandTag>
struct CollectOperatorAction {
  using local_operator_applied_to_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, OperandTag>;

  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ScalarFieldOperandTag,
            Requires<tmpl::list_contains_v<
                DbTagsList, local_operator_applied_to_operand_tag>> = nullptr>
  static void apply(
      db::DataBox<DbTagsList>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementId<1>& element_id,
      const Variables<tmpl::list<ScalarFieldOperandTag>>& Ap_global_data,
      const size_t broadcasting_mg_lev) noexcept {
    // FIXME: We're receiving broadcasts also from reductions over other
    // sections for some reason
    const size_t mg_lev =
        db::get<LinearSolver::multigrid::Tags::MultigridLevel>(box);
    if (mg_lev != broadcasting_mg_lev) {
      // Parallel::printf("Received broadcast on other MG level, ignoring.\n");
      return;
    }
    const size_t element_index = get_index(element_id);
    // This could be generalized to work on the Variables instead of the
    // Scalar, but it's only for the purpose of this test.
    const auto number_of_grid_points = get<LinearOperator>(box)[mg_lev][0].columns();
    const auto& Ap_global = get<ScalarFieldOperandTag>(Ap_global_data).get();
    DataVector Ap_local{number_of_grid_points};
    std::copy(Ap_global.begin() +
                  static_cast<int>(element_index * number_of_grid_points),
              Ap_global.begin() +
                  static_cast<int>((element_index + 1) * number_of_grid_points),
              Ap_local.begin());
    db::mutate<local_operator_applied_to_operand_tag>(
        make_not_null(&box),
        [&Ap_local, &number_of_grid_points](auto Ap) noexcept {
          *Ap = db::item_type<local_operator_applied_to_operand_tag>{
              number_of_grid_points};
          get(get<LinearSolver::Tags::OperatorAppliedTo<ScalarFieldOperandTag>>(
              *Ap)) = Ap_local;
        });
    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[element_id]
        .perform_algorithm(true);
  }
};

// Checks for the correct solution after the algorithm has terminated.
template <typename OptionsGroup>
struct TestResult {
  using const_global_cache_tags =
      tmpl::list<ExpectedResult, helpers::ExpectedConvergenceReason>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ElementId<1>& element_id,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t element_index = get_index(element_id);
    const auto& has_converged =
        get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             get<helpers::ExpectedConvergenceReason>(box));
    const auto& expected_result =
        gsl::at(get<ExpectedResult>(box), element_index);
    const auto& result = get<ScalarFieldTag>(box).get();
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    return {std::move(box), true};
  }
};

struct InitializeElement {
  using const_global_cache_tags = tmpl::list<Source>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementId<1>& element_id, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t element_index = get_index(element_id);
    const int mg_lev = element_id.segment_ids()[0].refinement_level();
    const auto& source = mg_lev == 1 ? db::item_type<sources_tag>(gsl::at(
                                           get<Source>(box), element_index))
                                     : db::item_type<sources_tag>{};
    const size_t num_points =
        db::get<domain::Tags::Mesh<1>>(box).number_of_grid_points();

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement, db::AddSimpleTags<fields_tag, sources_tag>>(
            std::move(box), db::item_type<fields_tag>{num_points, 0.}, source));
  }
};

namespace detail {

template <typename Preconditioner>
struct run_preconditioner {
  using type =
      tmpl::list<ComputeOperatorAction<typename Preconditioner::fields_tag>,
                 typename Preconditioner::prepare_solve,
                 ::Actions::RepeatUntil<
                     LinearSolver::Tags::HasConverged<
                         typename Preconditioner::options_group>,
                     tmpl::list<typename Preconditioner::prepare_step,
                                ComputeOperatorAction<
                                    typename Preconditioner::operand_tag>,
                                typename Preconditioner::perform_step>>>;
};
template <>
struct run_preconditioner<void> {
  using type = tmpl::list<>;
};

}  // namespace detail

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using initialization_actions = tmpl::list<
    dg::Actions::InitializeDomain<1>, InitializeElement,
    typename LinearSolverType::initialize_element,
    ComputeOperatorAction<fields_tag>,
    tmpl::type_from<helpers::detail::init_preconditioner<PreconditionerType>>,
    Initialization::Actions::RemoveOptionsAndTerminatePhase>;

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using register_actions = tmpl::list<typename LinearSolverType::register_element,
                                    typename LinearSolverType::prepare_solve,
                                    Parallel::Actions::TerminatePhase>;

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using solve_actions =
    tmpl::list<LinearSolver::Actions::TerminateIfConverged<
                   typename LinearSolverType::options_group>,
               typename LinearSolverType::prepare_step,
               tmpl::type_from<detail::run_preconditioner<PreconditionerType>>,
               ComputeOperatorAction<typename LinearSolverType::operand_tag>,
               typename LinearSolverType::perform_step>;

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using test_actions =
    tmpl::list<TestResult<typename LinearSolverType::options_group>>;

template <typename Metavariables>
using ElementArray = elliptic::DgElementArray<
    Metavariables,
    tmpl::list<
        Parallel::PhaseActions<typename Metavariables::Phase,
                               Metavariables::Phase::Initialization,
                               initialization_actions<Metavariables>>,
        Parallel::PhaseActions<typename Metavariables::Phase,
                               Metavariables::Phase::RegisterWithObserver,
                               register_actions<Metavariables>>,
        Parallel::PhaseActions<typename Metavariables::Phase,
                               Metavariables::Phase::PerformLinearSolve,
                               solve_actions<Metavariables>>,
        Parallel::PhaseActions<typename Metavariables::Phase,
                               Metavariables::Phase::TestResult,
                               test_actions<Metavariables>>>>;

template <typename Metavariables>
using component_list = tmpl::push_back<
    tmpl::append<typename Metavariables::linear_solver::component_list,
                 tmpl::type_from<helpers::detail::get_component_list<
                     typename Metavariables::preconditioner>>>,
    ElementArray<Metavariables>, observers::Observer<Metavariables>,
    observers::ObserverWriter<Metavariables>,
    helpers::OutputCleaner<Metavariables>>;

}  // namespace DistributedLinearSolverAlgorithmTestHelpers
