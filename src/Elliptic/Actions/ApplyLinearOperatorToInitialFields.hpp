// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Parallel/Tags.hpp"
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

template <typename FieldsTag, typename LinearOperandTag,
          typename ArraySectionIdTag>
struct CopyBackIntoFields;

template <typename FieldsTag, typename LinearOperandTag,
          typename ArraySectionIdTag>
struct CopyIntoLinearOperand {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, CopyIntoLinearOperand>::value;
    if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
      constexpr size_t skip_index =
          tmpl::index_of<ActionList,
                         CopyBackIntoFields<FieldsTag, LinearOperandTag,
                                            ArraySectionIdTag>>::value +
          1;
      return {std::move(box), false, skip_index};
    }

    db::mutate<LinearOperandTag>(
        make_not_null(&box),
        [](const auto linear_operand, const auto& fields) noexcept {
          *linear_operand = fields;
        },
        db::get<FieldsTag>(box));

    return {std::move(box), false, this_action_index + 1};
  }
};

template <typename FieldsTag, typename LinearOperandTag,
          typename ArraySectionIdTag>
struct CopyBackIntoFields {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
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

template <typename ApplyOperatorActions, typename FieldsTag,
          typename LinearOperandTag, typename ArraySectionIdTag>
using apply_linear_operator_to_initial_fields = tmpl::list<
    detail::CopyIntoLinearOperand<FieldsTag, LinearOperandTag,
                                  ArraySectionIdTag>,
    ApplyOperatorActions,
    detail::CopyBackIntoFields<FieldsTag, LinearOperandTag, ArraySectionIdTag>>;
}  // namespace elliptic::Actions
