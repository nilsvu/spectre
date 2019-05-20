// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"

#include "Parallel/Printf.hpp"

namespace db {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Mutate the DataBox tags in `TagsList` according to the `data`.
 *
 * DataBox changes:
 * - Modifies:
 *   - All tags in `TagsList`
 */
template <typename TagsList>
struct SetData {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      tuples::tagged_tuple_from_typelist<TagsList> data) noexcept {
    Parallel::printf("%s SetData:\n%s\n", array_index, data);
    tmpl::for_each<TagsList>([&box, &data ](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      db::mutate<tag>(
          make_not_null(&box), [&data](const gsl::not_null<db::item_type<tag>*>
                                           value) noexcept {
            *value = std::move(get<tag>(data));
          });
    });
  }
};

}  // namespace Actions
}  // namespace db
