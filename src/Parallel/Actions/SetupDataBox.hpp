// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
/// \cond
struct SetupDataBox;
/// \endcond

namespace detail {
CREATE_HAS_TYPE_ALIAS(simple_tags)
CREATE_HAS_TYPE_ALIAS_V(simple_tags)
CREATE_HAS_TYPE_ALIAS(compute_tags)
CREATE_HAS_TYPE_ALIAS_V(compute_tags)

template <typename Action, typename enable = std::bool_constant<true>>
struct optional_simple_tags;

template <typename Action>
struct optional_simple_tags<
    Action, std::bool_constant<detail::has_simple_tags_v<Action>>> {
  using type = typename Action::simple_tags;
};

template <typename Action>
struct optional_simple_tags<
    Action, std::bool_constant<not detail::has_simple_tags_v<Action>>> {
  using type = tmpl::list<>;
};

template <typename Action, typename enable = std::bool_constant<true>>
struct optional_compute_tags;

template <typename Action>
struct optional_compute_tags<
    Action, std::bool_constant<detail::has_compute_tags_v<Action>>> {
  using type = typename Action::compute_tags;
};

template <typename Action>
struct optional_compute_tags<
    Action, std::bool_constant<not detail::has_compute_tags_v<Action>>> {
  using type = tmpl::list<>;
};

template <typename ActionList>
struct get_action_list_simple_tags;

template <typename... ActionList>
struct get_action_list_simple_tags<tmpl::list<ActionList...>> {
  using type = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::list<typename optional_simple_tags<ActionList>::type...>>>;
};

template <typename ActionList>
struct get_action_list_compute_tags;

template <typename... ActionList>
struct get_action_list_compute_tags<tmpl::list<ActionList...>> {
  using type = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::list<typename optional_compute_tags<ActionList>::type...>>>;
};

template <typename DbTags, typename... simple_tags, typename... compute_tags>
static auto merge_into_databox_helper(
    db::DataBox<DbTags>&& box, tmpl::list<simple_tags...> /*meta*/,
    tmpl::list<compute_tags...> /*meta*/) noexcept {
  if constexpr (not tmpl2::flat_any_v<
                    tmpl::list_contains_v<DbTags, simple_tags>...> and
                not tmpl2::flat_any_v<
                    tmpl::list_contains_v<DbTags, compute_tags>...>) {
    return db::create_from<db::RemoveTags<>, db::AddSimpleTags<simple_tags...>,
                           db::AddComputeTags<compute_tags...>>(
        std::move(box), typename simple_tags::type{}...);
  } else {
    ERROR(
        "The SetupDataBox action must be called only once, and has either "
        "been called more than once or one or more of the tags it is adding "
        "are already present.");
    return std::move(box);
  }
}
}  // namespace detail

/*!
 * \brief Merge into the \ref DataBoxGroup the collection of tags requested for
 * initialization in any of the actions.
 *
 * \details This action adds all of the simple tags given in the `simple_tags`
 * type lists in each of the other actions in the current phase, and all of the
 * compute tags given in the `compute_tags` type lists. If an action does not
 * give either of the type lists, it is treated as an empty type list.
 *
 * To prevent the proliferation of many \ref DataBoxGroup types, which can
 * drastically slow compile times, it is preferable to use only this action to
 * add tags to the \ref DataBoxGroup, and place this action at the start of the
 * `Initialization` phase action list. The rest of the initialization actions
 * should specify `simple_tags` and `compute_tags`, and assign initial values to
 * those tags, but not add those tags into the \ref DataBoxGroup.
 */
struct SetupDataBox {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using action_list_simple_tags =
        typename detail::get_action_list_simple_tags<ActionList>::type;
    using action_list_compute_tags =
        typename detail::get_action_list_compute_tags<ActionList>::type;
    // grab the simple_tags, compute_tags, mutate the databox, creating
    // default-constructed objects.
    return std::make_tuple(detail::merge_into_databox_helper(
        std::move(box), action_list_simple_tags{}, action_list_compute_tags{}));
  }
};
}  // namespace Actions
