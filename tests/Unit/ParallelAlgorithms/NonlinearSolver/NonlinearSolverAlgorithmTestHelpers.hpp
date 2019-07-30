// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/NumericalAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"

namespace NonlinearSolverAlgorithmTestHelpers {

struct LinearOperator : db::SimpleTag {
  static constexpr OptionString help = "The linear operator A to invert.";
  using type = DenseMatrix<double>;
  using option_tags = tmpl::list<LinearOperator>;
  static type create_from_options(const type& option) { return option; }
  static std::string name() noexcept { return "LinearOperator"; }
};
struct Source : db::SimpleTag {
  static constexpr OptionString help = "The source b in the equation Ax=b.";
  using type = DenseVector<double>;
  using option_tags = tmpl::list<Source>;
  static type create_from_options(const type& option) { return option; }
  static std::string name() noexcept { return "Source"; }
};
struct InitialGuess : db::SimpleTag {
  static constexpr OptionString help = "The initial guess for the vector x.";
  using type = DenseVector<double>;
  using option_tags = tmpl::list<InitialGuess>;
  static type create_from_options(const type& option) { return option; }
  static std::string name() noexcept { return "InitialGuess"; }
};
struct ExpectedResult : db::SimpleTag {
  static constexpr OptionString help = "The solution x in the equation Ax=b";
  using type = DenseVector<double>;
  using option_tags = tmpl::list<ExpectedResult>;
  static type create_from_options(const type& option) { return option; }
  static std::string name() noexcept { return "ExpectedResult"; }
};

// The vector `x` we want to solve for
struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using fields_tag = VectorTag;
using nonlinear_source_tag = ::Tags::FixedSource<fields_tag>;
using nonlinear_operator_tag =
    NonlinearSolver::Tags::OperatorAppliedTo<fields_tag>;
using correction_tag = NonlinearSolver::Tags::Correction<fields_tag>;
using linear_source_tag = ::Tags::FixedSource<correction_tag>;
using linear_operator_correction_tag =
    LinearSolver::Tags::OperatorAppliedTo<correction_tag>;
using linear_operand_tag = LinearSolver::Tags::Operand<correction_tag>;
using linear_operator_tag =
    LinearSolver::Tags::OperatorAppliedTo<linear_operand_tag>;

struct BuildNonlinearOperator {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const int /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) noexcept {
    db::mutate<nonlinear_operator_tag>(
        make_not_null(&box),
        [](const auto Ax, const auto& A, const auto& x) noexcept {
          *Ax = A * x;
        },
        get<LinearOperator>(cache), get<VectorTag>(box));
    return {std::move(box)};
  }
};

struct BuildLinearOperator {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const int /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) noexcept {
    db::mutate<linear_operator_tag>(make_not_null(&box),
                                    [](const auto Ap, const auto& A,
                                       const auto& p) noexcept { *Ap = A * p; },
                                    get<LinearOperator>(cache),
                                    get<linear_operand_tag>(box));
    return {std::move(box)};
  }
};

// Checks for the correct solution after the algorithm has terminated.
struct TestResult {
  using const_global_cache_tags = tmpl::list<ExpectedResult>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const int /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& has_converged = get<NonlinearSolver::Tags::HasConverged>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             Convergence::Reason::AbsoluteResidual);
    const auto& result = get<VectorTag>(box);
    const auto& expected_result = get<ExpectedResult>(cache);
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    return {std::move(box), true};
  }
};

struct InitializeElement {
  using const_global_cache_tags =
      tmpl::list<LinearOperator, Source, InitialGuess>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int /*array_index*/, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& A = get<LinearOperator>(cache);
    const auto& b = get<Source>(cache);
    const auto& x0 = get<InitialGuess>(cache);

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<fields_tag, nonlinear_source_tag,
                              nonlinear_operator_tag, correction_tag,
                              linear_operand_tag, linear_operator_tag>>(
            std::move(box), DenseVector<double>(x0), DenseVector<double>(b),
            DenseVector<double>(A * x0),
            make_with_value<db::item_type<correction_tag>>(
                x0, std::numeric_limits<double>::signaling_NaN()),
            make_with_value<db::item_type<linear_operand_tag>>(
                x0, std::numeric_limits<double>::signaling_NaN()),
            make_with_value<db::item_type<linear_operator_tag>>(
                x0, std::numeric_limits<double>::signaling_NaN())));
  }
};

template <typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using array_index = int;
  using metavariables = Metavariables;

  using nonlinear_solver = typename Metavariables::nonlinear_solver;
  using linear_solver = typename Metavariables::linear_solver;

  using solve_linearized_system = tmpl::list<
      typename linear_solver::prepare_solve,
      ::Actions::WhileNot<
          LinearSolver::Tags::HasConverged,
          tmpl::list<typename linear_solver::prepare_step, BuildLinearOperator,
                     typename linear_solver::perform_step>>>;

  /// [action_list]
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::flatten<tmpl::list<
              InitializeElement, typename nonlinear_solver::initialize_element,
              typename linear_solver::initialize_element,
              typename nonlinear_solver::prepare_solve, BuildNonlinearOperator,
              typename nonlinear_solver::update_residual,
              Parallel::Actions::TerminatePhase>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Solve,
          tmpl::flatten<tmpl::list<
              NonlinearSolver::Actions::TerminateIfConverged,
              typename nonlinear_solver::prepare_step, solve_linearized_system,
              ::Actions::WhileNot<
                  NonlinearSolver::Tags::GlobalizationHasConverged,
                  tmpl::list<typename nonlinear_solver::perform_step,
                             BuildNonlinearOperator,
                             typename nonlinear_solver::update_residual>>>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TestResult,
                             tmpl::list<TestResult>>>;
  /// [action_list]

  using array_allocation_tags = tmpl::list<>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;

  static void allocate_array(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept {
    auto& local_component = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    local_component[0].insert(global_cache, initialization_items, 0);
    local_component.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_component = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    local_component.start_phase(next_phase);
  }
};

template <typename Metavariables>
using OutputCleaner =
    LinearSolverAlgorithmTestHelpers::OutputCleaner<Metavariables>;

}  // namespace NonlinearSolverAlgorithmTestHelpers
