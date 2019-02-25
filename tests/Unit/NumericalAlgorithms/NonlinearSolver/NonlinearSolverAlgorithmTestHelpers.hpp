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
#include "NumericalAlgorithms/LinearSolver/Actions/ReinitializeIfConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/NonlinearSolver/Actions/TerminateIfConverged.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/NumericalAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"

namespace NonlinearSolverAlgorithmTestHelpers {

struct LinearOperator {
  static constexpr OptionString help = "The linear operator A to invert.";
  using type = DenseMatrix<double>;
};
struct Source {
  static constexpr OptionString help = "The source b in the equation Ax=b.";
  using type = DenseVector<double>;
};
struct InitialGuess {
  static constexpr OptionString help = "The initial guess for the vector x.";
  using type = DenseVector<double>;
};
struct ExpectedResult {
  static constexpr OptionString help = "The solution x in the equation Ax=b";
  using type = DenseVector<double>;
};

// The vector `x` we want to solve for
struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using nonlinear_source_tag = ::Tags::Source<VectorTag>;
using nonlinear_operator_tag =
    NonlinearSolver::Tags::OperatorAppliedTo<VectorTag>;
using correction_tag = NonlinearSolver::Tags::Correction<VectorTag>;
using linear_source_tag = ::Tags::Source<correction_tag>;
using linear_operator_correction_tag =
    LinearSolver::Tags::OperatorAppliedTo<correction_tag>;
using linear_operand_tag = LinearSolver::Tags::Operand<correction_tag>;
using linear_operator_tag =
    LinearSolver::Tags::OperatorAppliedTo<linear_operand_tag>;

struct ComputeNonlinearOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int /*array_index*/, const ActionList /*meta*/,
                    const ParallelComponent* const /*component*/) noexcept {
    if (not get<LinearSolver::Tags::HasConverged>(box)) {
      return std::forward_as_tuple(std::move(box));
    }
    db::mutate<nonlinear_operator_tag>(
        make_not_null(&box),
        [](const auto Ax, const auto& A, const auto& x) noexcept {
          *Ax = A * x;
        },
        get<LinearOperator>(cache), get<VectorTag>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

struct ComputeLinearOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int /*array_index*/, const ActionList /*meta*/,
                    const ParallelComponent* const /*component*/) noexcept {
    db::mutate<linear_operator_tag>(make_not_null(&box),
                                    [](const auto Ap, const auto& A,
                                       const auto& p) noexcept { *Ap = A * p; },
                                    get<LinearOperator>(cache),
                                    get<linear_operand_tag>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

// Checks for the correct solution after the algorithm has terminated.
struct TestResult {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& box,
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
  }
};

struct InitializeElement {
  template <typename Metavariables>
  using return_tag_list = tmpl::append<
      tmpl::list<VectorTag, nonlinear_source_tag, nonlinear_operator_tag,
                 correction_tag, linear_operand_tag, linear_operator_tag>,
      typename Metavariables::nonlinear_solver::tags::simple_tags,
      typename Metavariables::nonlinear_solver::tags::compute_tags,
      typename Metavariables::linear_solver::tags::simple_tags,
      typename Metavariables::linear_solver::tags::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(
      const db::DataBox<tmpl::list<>>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const int array_index, const ActionList /*meta*/,
      const ParallelComponent* const parallel_component_meta) noexcept {
    const auto& A = get<LinearOperator>(cache);
    const auto& b = get<Source>(cache);
    const auto& x0 = get<InitialGuess>(cache);

    auto box = db::create<db::AddSimpleTags<
        tmpl::list<VectorTag, nonlinear_source_tag, nonlinear_operator_tag,
                   correction_tag, linear_operand_tag, linear_operator_tag>>>(
        x0, b, DenseVector<double>(A * x0),
        make_with_value<db::item_type<correction_tag>>(
            x0, std::numeric_limits<double>::signaling_NaN()),
        make_with_value<db::item_type<linear_operand_tag>>(
            x0, std::numeric_limits<double>::signaling_NaN()),
        make_with_value<db::item_type<linear_operand_tag>>(
            x0, std::numeric_limits<double>::signaling_NaN()));
    auto nonlinear_solver_box =
        Metavariables::nonlinear_solver::tags::initialize(
            std::move(box), cache, array_index, parallel_component_meta);
    auto linear_solver_box = Metavariables::linear_solver::tags::initialize(
        std::move(nonlinear_solver_box), cache, array_index,
        parallel_component_meta);
    return std::make_tuple(std::move(linear_solver_box));
  }
};  // namespace

template <typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  // In each step of the algorithm we must provide A(p). The linear solver then
  // takes care of updating x and p, as well as the internal variables r, its
  // magnitude and the iteration step number.
  /// [action_list]
  using action_list = tmpl::list<
      NonlinearSolver::Actions::TerminateIfConverged,
      LinearSolver::Actions::ReinitializeIfConverged,
      ComputeLinearOperatorAction,
      typename Metavariables::linear_solver::perform_step,
      typename Metavariables::nonlinear_solver::perform_step,
      ComputeNonlinearOperatorAction,
      typename Metavariables::nonlinear_solver::prepare_linear_solve>;
  /// [action_list]
  using initial_databox = db::compute_databox_type<
      typename InitializeElement::return_tag_list<Metavariables>>;
  using options = tmpl::list<>;
  using const_global_cache_tag_list =
      tmpl::list<LinearOperator, Source, InitialGuess, ExpectedResult>;
  using array_index = int;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& array_proxy = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    array_proxy[0].insert(global_cache, 0);
    array_proxy.doneInserting();

    Parallel::simple_action<InitializeElement>(array_proxy);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto array_proxy = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    switch (next_phase) {
      case Metavariables::Phase::PerformLinearSolve:
        array_proxy.perform_algorithm();
        break;
      case Metavariables::Phase::TestResult:
        Parallel::simple_action<TestResult>(array_proxy);
        break;
      case Metavariables::Phase::CleanOutput:
        break;
      default:
        ERROR(
            "The Metavariables is expected to have the following Phases: "
            "Initialization, PerformLinearSolve, TestResult, Exit");
    }
  }
};

template <typename Metavariables>
using OutputCleaner =
    LinearSolverAlgorithmTestHelpers::OutputCleaner<Metavariables>;

struct System {
  using nonlinear_fields_tag = VectorTag;
  using fields_tag = correction_tag;
  using compute_nonlinear_operator_action = ComputeNonlinearOperatorAction;
};

}  // namespace NonlinearSolverAlgorithmTestHelpers
