// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Mutate the DataBox tags in `TagsList` according to the `data`.
 *
 * An example use case for this action is as the callback for the
 * `importers::ThreadedActions::ReadVolumeData`.
 *
 * DataBox changes:
 * - Modifies:
 *   - All tags in `TagsList`
 */
template <typename TagsList>
struct SetData;

/// \cond
template <typename... Tags>
struct SetData<tmpl::list<Tags...>> {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<std::conjunction_v<db::tag_is_retrievable<Tags, DataBox>...>> =
          nullptr>
  static void apply(DataBox& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    tuples::TaggedTuple<Tags...> data) noexcept {
    tmpl::for_each<tmpl::list<Tags...>>([&box, &data](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      db::mutate<tag>(make_not_null(&box), [&data](const auto value) noexcept {
        *value = std::move(tuples::get<tag>(data));
      });
    });
  }
};
/// \endcond

template <typename SourceTag, typename TargetTag>
struct Copy {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<TargetTag>(
        make_not_null(&box),
        [](const auto target, const auto& source) noexcept {
          *target = source;
        },
        db::get<SourceTag>(box));
    return {std::move(box)};
  }
};

}  // namespace Actions
