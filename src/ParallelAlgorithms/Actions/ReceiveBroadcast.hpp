// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Tags {

template <typename Tag, typename TemporalIdTag>
struct BroadcastInbox
    : Parallel::InboxInserters::Value<BroadcastInbox<Tag, TemporalIdTag>> {
  using temporal_id = typename TemporalIdTag::type;
  using type = std::map<temporal_id, typename Tag::type>;
  using tag = Tag;
};

}  // namespace Tags

namespace Actions {

template <typename TemporalIdTag, typename BroadcastTags>
struct ReceiveBroadcast {
  using simple_tags = BroadcastTags;
  using inbox_tags =
      db::wrap_tags_in<::Tags::BroadcastInbox, BroadcastTags, TemporalIdTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    bool isready = true;
    tmpl::for_each<BroadcastTags>([&box, &inboxes, &isready](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      const auto& inbox =
          get<::Tags::BroadcastInbox<tag, TemporalIdTag>>(inboxes);
      const size_t temporal_id = get<TemporalIdTag>(box);
      if (inbox.find(temporal_id) == inbox.end()) {
        isready = false;
      }
    });
    if (not isready) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    tmpl::for_each<BroadcastTags>([&box, &inboxes](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      auto& inbox = get<::Tags::BroadcastInbox<tag, TemporalIdTag>>(inboxes);
      const auto temporal_id_and_value = inbox.find(get<TemporalIdTag>(box));
      db::mutate<tag>(
          make_not_null(&box),
          [](const auto local_value, const auto& value) {
            *local_value = value;
          },
          temporal_id_and_value->second);
      inbox.erase(temporal_id_and_value);
    });
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace Actions
