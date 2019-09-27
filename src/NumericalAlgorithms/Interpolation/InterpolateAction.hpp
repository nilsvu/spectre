// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/ElementId.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <size_t VolumeDim>
struct Mesh;
template <typename TagsList>
struct Variables;
}  // namespace Tags
template <size_t Dim>
class Mesh;
template <size_t VolumeDim>
class ElementIndex;
/// \endcond

namespace intrp {
namespace Actions {

template <typename InterpolationTargetTag,
          typename BroadcastTags =
              typename InterpolationTargetTag::broadcast_tags>
struct InitializeBroadcast {};

template <typename InterpolationTargetTag, typename... BroadcastTags>
struct InitializeBroadcast<InterpolationTargetTag,
                           tmpl::list<BroadcastTags...>> {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeBroadcast,
                                             tmpl::list<BroadcastTags...>>(
            std::move(box), db::item_type<BroadcastTags>{}...));
  }
};

template <typename TemporalId, typename Tensors>
struct Interpolate {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t VolumeDim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<VolumeDim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& mesh = get<::Tags::Mesh<VolumeDim>>(box);
    Variables<Tensors> interp_vars{mesh.number_of_grid_points()};
    tmpl::for_each<Tensors>(
        [&interp_vars, &box ](const auto tensor_tag_v) noexcept {
          using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
          get<tensor_tag>(interp_vars) = get<tensor_tag>(box);
        });

    // Send volume data to the Interpolator to trigger interpolation.
    auto& interpolator =
        *::Parallel::get_parallel_component<Interpolator<Metavariables>>(cache)
             .ckLocalBranch();
    Parallel::simple_action<InterpolatorReceiveVolumeData>(
        interpolator, get<TemporalId>(box), ElementId<VolumeDim>(array_index),
        mesh, interp_vars);

    return {std::move(box)};
  }
};

template <typename TemporalIdTag, typename BroadcastTags>
struct ReceiveBroadcast {
  using inbox_tags =
      db::wrap_tags_in<Tags::Broadcast, BroadcastTags, TemporalIdTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t VolumeDim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<VolumeDim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    tmpl::for_each<BroadcastTags>([&box, &inboxes](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      auto& inbox = get<Tags::Broadcast<tag, TemporalIdTag>>(inboxes);
      const auto temporal_id_and_value = inbox.find(get<TemporalIdTag>(box));
      db::mutate<tag>(make_not_null(&box),
                      [](const gsl::not_null<db::item_type<tag>*> local_value,
                         const db::item_type<tag>& value) noexcept {
                        *local_value = value;
                      },
                      temporal_id_and_value->second);
      inbox.erase(temporal_id_and_value);
    });
    return {std::move(box)};
  }
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    bool isready = true;
    tmpl::for_each<BroadcastTags>(
        [&box, &inboxes, &isready ](auto tag_v) noexcept {
          using tag = tmpl::type_from<decltype(tag_v)>;
          const auto& inbox = get<Tags::Broadcast<tag, TemporalIdTag>>(inboxes);
          if (inbox.find(get<TemporalIdTag>(box)) == inbox.end()) {
            isready = false;
          }
        });
    return isready;
  }
};

template <typename InterpolationTargetTags>
using make_initialize_broadcast_actions =
    tmpl::transform<InterpolationTargetTags,
                    tmpl::bind<InitializeBroadcast, tmpl::_1>>;

}  // namespace Actions
}  // namespace intrp
