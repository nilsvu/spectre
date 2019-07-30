// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

template <typename Metavariables, typename FieldsTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
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
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        ::Tags::NextCompute<NonlinearSolver::Tags::IterationId>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<NonlinearSolver::Tags::IterationId,
                              linear_source_tag, linear_operator_tag,
                              NonlinearSolver::Tags::HasConverged>,
            compute_tags>(
            // The `PrepareSolve` action populates these tags with initial
            // values
            std::move(box), std::numeric_limits<size_t>::max(),
            db::item_type<linear_source_tag>{},
            db::item_type<linear_operator_tag>{},
            db::item_type<NonlinearSolver::Tags::HasConverged>{}));
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
