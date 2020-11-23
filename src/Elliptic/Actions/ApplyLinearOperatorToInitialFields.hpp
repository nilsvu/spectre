// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

namespace detail {

template <typename FieldsTag, typename LinearOperandTag>
struct CopyBackIntoFields;

template <typename FieldsTag, typename LinearOperandTag>
struct CopyIntoLinearOperand {
  static_assert(
      not std::is_same_v<FieldsTag, LinearOperandTag>,
      "The FieldsTag and the LinearOperandTag are the same. You don't need "
      "apply_linear_operator_to_initial_fields but you can just add the linear "
      "operator actions to the action list directly.");

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearOperandTag>(
        make_not_null(&box),
        [](const auto linear_operand, const auto& fields) noexcept {
          *linear_operand = fields;
        },
        db::get<FieldsTag>(box));
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename LinearOperandTag>
struct CopyBackIntoFields {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag>>(
        make_not_null(&box),
        [](const auto operator_applied_to_fields,
           const auto& operator_applied_to_linear_operand) noexcept {
          *operator_applied_to_fields = operator_applied_to_linear_operand;
        },
        db::get<db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                                   LinearOperandTag>>(box));
    return {std::move(box)};
  }
};

}  // namespace detail

/// Apply the `ApplyOperatorActions` to the `FieldsTag` by copying the data into
/// the `LinearOperandTag` and back. This is necessary when the
/// `LinearOperandTag` and the `FieldsTag` are not the same, and thus the
/// `ApplyOperatorActions` work with the `LinearOperandTag`.
///
/// \warning The data in `LinearOperandTag` and
/// `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
/// LinearOperandTag>` will be overwritten, in addition to the result variable
/// `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag`.
template <typename ApplyOperatorActions, typename FieldsTag,
          typename LinearOperandTag>
using apply_linear_operator_to_initial_fields =
    tmpl::list<detail::CopyIntoLinearOperand<FieldsTag, LinearOperandTag>,
               ApplyOperatorActions,
               detail::CopyBackIntoFields<FieldsTag, LinearOperandTag>>;
}  // namespace elliptic::Actions
