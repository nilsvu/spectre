// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Apply the function `Mutator::apply` to the DataBox
 *
 * The function `Mutator::apply` is invoked with the `Mutator::argument_tags`.
 * The result of this computation is stored in the `Mutator::return_tags`.
 *
 * Uses:
 * - DataBox:
 *   - All elements in `Mutator::argument_tags`
 *
 * DataBox changes:
 * - Modifies:
 *   - All elements in `Mutator::return_tags`
 */
template <typename Mutator, typename = std::nullptr_t>
struct MutateApply;

template <typename Mutator>
struct MutateApply<Mutator, Requires<not db::is_tag_v<Mutator>>> {
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_action<Mutator>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<DataBox&&> apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate_apply<Mutator>(make_not_null(&box));
    return {std::move(box)};
  }
};

template <typename Mutator>
struct MutateApply<Mutator, Requires<db::is_tag_v<Mutator>>> {
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_action<Mutator>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<DataBox&&> apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& mutator = db::get<Mutator>(box);
    db::mutate_apply(mutator, make_not_null(&box));
    return {std::move(box)};
  }
};
}  // namespace Actions
