// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver {
namespace newton_raphson_detail {
template <typename Metavariables, typename FieldsTag>
struct ResidualMonitor;
}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

struct PrepareSolve {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<NonlinearSolver::Tags::IterationId,
               NonlinearSolver::Tags::GlobalizationIterationId>(
        make_not_null(&box), [](const gsl::not_null<size_t*> iteration_id,
                                const gsl::not_null<size_t*>
                                    globalization_iteration_id) noexcept {
          *iteration_id = 0;
          *globalization_iteration_id = 0;
        });
    return {std::move(box)};
  }
};

struct PrepareStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<NonlinearSolver::Tags::IterationId>(
        make_not_null(&box),
        [](const gsl::not_null<
               db::item_type<NonlinearSolver::Tags::IterationId>*>
               iteration_id,
           const db::item_type<
               ::Tags::Next<NonlinearSolver::Tags::IterationId>>&
               next_iteration_id) noexcept {
          *iteration_id = next_iteration_id;
        },
        get<::Tags::Next<NonlinearSolver::Tags::IterationId>>(box));
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename GlobalizationStrategy>
struct UpdateResidual {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_source_tag =
      db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using nonlinear_operator_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using linear_source_tag =
      db::add_tag_prefix<::Tags::FixedSource, correction_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ASSERT(get<NonlinearSolver::Tags::IterationId>(box) !=
               std::numeric_limits<size_t>::max(),
           "Nonlinear solve iteration ID is at initial state. Did you forget "
           "to invoke 'PrepareStep'?");

    db::mutate<linear_source_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<linear_source_tag>*> linear_source,
           const db::item_type<nonlinear_source_tag>& nonlinear_source,
           const db::item_type<nonlinear_operator_tag>& nonlinear_operator) {
          // Compute new nonlinear residual, which sources the next linear
          // solve. The nonlinear source b stays the same, but need to apply the
          // nonlinear operator to the new field values. We assume this has been
          // done at this point.
          *linear_source = nonlinear_source - nonlinear_operator;
        },
        get<nonlinear_source_tag>(box), get<nonlinear_operator_tag>(box));

    Parallel::contribute_to_reduction<UpdateResidualMagnitude<
        FieldsTag, GlobalizationStrategy, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            get<Tags::TemporalId>(box), get<Tags::IterationId>(box),
            get<Tags::GlobalizationIterationId>(box),
            get<Tags::StepLength>(box),
            LinearSolver::inner_product(get<linear_source_tag>(box),
                                        get<linear_source_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag>>(cache));

    // Terminate algorithm for now. The reduction will be broadcast to the
    // action below, which is responsible for restarting the algorithm.
    return {std::move(box), true};
  }
};

template <typename FieldsTag>
struct UpdateHasConverged {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_source_tag =
      db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using linear_operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, correction_tag>;

 public:
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<
          db::tag_is_retrievable_v<NonlinearSolver::Tags::HasConverged,
                                   DataBox> and
          db::tag_is_retrievable_v<
              NonlinearSolver::Tags::GlobalizationHasConverged, DataBox> and
          db::tag_is_retrievable_v<correction_tag, DataBox> and
          db::tag_is_retrievable_v<nonlinear_source_tag, DataBox> and
          db::tag_is_retrievable_v<linear_operator_tag, DataBox>> = nullptr>
  static void apply(DataBox& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const db::item_type<NonlinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<NonlinearSolver::Tags::HasConverged,
               NonlinearSolver::Tags::GlobalizationHasConverged, correction_tag,
               linear_operator_tag>(
        make_not_null(&box),
        [&has_converged](
            const gsl::not_null<
                db::item_type<NonlinearSolver::Tags::HasConverged>*>
                local_has_converged,
            const gsl::not_null<bool*> globalization_has_converged,
            const gsl::not_null<db::item_type<correction_tag>*> correction,
            const gsl::not_null<db::item_type<linear_operator_tag>*>
                linear_operator,
            const db::item_type<nonlinear_source_tag>& used_for_size) noexcept {
          *local_has_converged = has_converged;
          *globalization_has_converged = true;
          // Begin the linear solve with a zero initial guess
          *correction =
              make_with_value<db::item_type<correction_tag>>(used_for_size, 0.);
          *linear_operator = *correction;
        },
        get<nonlinear_source_tag>(box));

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

template <typename FieldsTag>
struct PerformStep {
 private:
  using fields_tag = FieldsTag;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Apply the correction that the linear solve has determined to improve
    // the nonlinear solution
    db::mutate<fields_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<fields_tag>*> fields,
           const db::item_type<correction_tag>& correction,
           const double& step_length) { *fields += step_length * correction; },
        get<correction_tag>(box), get<NonlinearSolver::Tags::StepLength>(box));

    return {std::move(box)};
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
