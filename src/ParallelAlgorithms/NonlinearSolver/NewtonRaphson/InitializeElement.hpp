// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace newton_raphson_detail {
template <typename Metavariables, typename FieldsTag>
struct ResidualMonitor;
template <typename FieldsTag, typename BroadcastTarget>
struct InitializeResidual;
}  // namespace newton_raphson_detail
}  // namespace LinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

template <typename Metavariables, typename FieldsTag,
          Initialization::MergePolicy MergePolicy =
              Initialization::MergePolicy::Error>
struct InitializeElement {
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
  using linear_operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, correction_tag>;

 public:
  template <typename DataBox, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags =
        db::AddSimpleTags<NonlinearSolver::Tags::IterationId, linear_source_tag,
                          linear_operator_tag,
                          NonlinearSolver::Tags::HasConverged>;
    using compute_tags = db::AddComputeTags<>;

    // Compute nonlinear residual. It sources the linear solve for the
    // correction, so directly store as such in the DataBox.
    auto linear_source = db::item_type<linear_source_tag>(
        get<nonlinear_source_tag>(box) - get<nonlinear_operator_tag>(box));

    // Always start with a zero initial guess for the correction
    db::mutate<correction_tag>(
        make_not_null(&box), [&linear_source](const gsl::not_null<
                                              db::item_type<correction_tag>*>
                                                  correction) noexcept {
          *correction =
              make_with_value<db::item_type<correction_tag>>(linear_source, 0.);
        });

    // Since the correction is zero, so is the linear operator applied to it
    auto linear_operator =
        make_with_value<db::item_type<linear_operator_tag>>(linear_source, 0.);

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    Parallel::contribute_to_reduction<newton_raphson_detail::InitializeResidual<
        FieldsTag, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            LinearSolver::inner_product(linear_source, linear_source)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag>>(cache));

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeElement, simple_tags,
                                             compute_tags, MergePolicy>(
            std::move(box),
            db::item_type<NonlinearSolver::Tags::IterationId>{0},
            std::move(linear_source), std::move(linear_operator),
            db::item_type<NonlinearSolver::Tags::HasConverged>{}),
        // Terminate algorithm for now. The reduction will be broadcast to the
        // next action which is responsible for restarting the algorithm.
        true);
  }
};

template <typename FieldsTag>
struct InitializeHasConverged {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<
                NonlinearSolver::Tags::HasConverged, DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const db::item_type<NonlinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<NonlinearSolver::Tags::HasConverged>(
        make_not_null(&box), [&has_converged](
                                 const gsl::not_null<db::item_type<
                                     NonlinearSolver::Tags::HasConverged>*>
                                     local_has_converged) noexcept {
          *local_has_converged = has_converged;
        });

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
