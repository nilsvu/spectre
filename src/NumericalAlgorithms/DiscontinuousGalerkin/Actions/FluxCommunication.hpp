// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Printf.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Next;
}  // namespace Tags
/// \endcond

namespace dg {

/// The inbox tag for flux communication.
template <typename BoundaryScheme>
struct FluxesInboxTag {
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id =
      db::const_item_type<typename BoundaryScheme::temporal_id_tag>;
  using type = std::map<
      temporal_id,
      FixedHashMap<
          maximum_number_of_neighbors(volume_dim),
          std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
          std::pair<temporal_id, typename BoundaryScheme::BoundaryData>,
          boost::hash<
              std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>>;
};

namespace Actions {

template <typename InboxTag, size_t Dim, typename TemporalIdType,
          typename... InboxTags>
bool has_received_from_all_neighbors(
    const TemporalIdType& temporal_id, const Element<Dim>& element,
    const tuples::TaggedTuple<InboxTags...>& inboxes) noexcept {
  if (element.number_of_neighbors() == 0) {
    return true;
  }
  const auto& inbox = tuples::get<InboxTag>(inboxes);
  const auto temporal_received = inbox.find(temporal_id);
  if (temporal_received == inbox.end()) {
    return false;
  }
  const auto& received_neighbor_data = temporal_received->second;
  for (const auto& direction_and_neighbors : element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    for (const auto& neighbor : direction_and_neighbors.second) {
      const auto neighbor_received =
          received_neighbor_data.find(std::make_pair(direction, neighbor));
      if (neighbor_received == received_neighbor_data.end()) {
        return false;
      }
    }
  }
  return true;
}

/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Receive boundary data needed for fluxes from neighbors.
 *
 * Uses:
 * - DataBox:
 *   - BoundaryScheme::temporal_id
 *   - Tags::Next<BoundaryScheme::temporal_id_tag>
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - Tags::Mortars<Tags::Next<BoundaryScheme::temporal_id_tag>, volume_dim>
 *   - Tags::Mortars<BoundaryScheme::mortar_data_tag, volume_dim>
 *
 * \see SendDataForFluxes
 */
template <typename BoundaryScheme>
struct ReceiveDataForFluxes {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using fluxes_inbox_tag = dg::FluxesInboxTag<BoundaryScheme>;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;

 public:
  using inbox_tags = tmpl::list<fluxes_inbox_tag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox = tuples::get<fluxes_inbox_tag>(inboxes);
    const auto& temporal_id = get<temporal_id_tag>(box);
    const auto temporal_received = inbox.find(temporal_id);

    // Parallel::printf("%s Received data at %s.\n", element_index,
    // temporal_id);

    db::mutate<all_mortar_data_tag>(
        make_not_null(&box),
        [&temporal_received](
            const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                mortar_data) noexcept {
          for (auto& received_mortar_data : temporal_received->second) {
            const auto& mortar_id = received_mortar_data.first;
            mortar_data->at(mortar_id).remote_insert(
                temporal_received->first,
                std::move(received_mortar_data.second.second));
          }
        });
    inbox.erase(temporal_received);

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    return ::dg::Actions::has_received_from_all_neighbors<fluxes_inbox_tag>(
        get<temporal_id_tag>(box), get<::Tags::Element<volume_dim>>(box),
        inboxes);
  }
};

/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Send local boundary data needed for fluxes to neighbors.
 *
 * With:
 * - `Interface<Tag> =
 *   Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>`
 *
 * Uses:
 * - DataBox:
 *   - Tags::Element<volume_dim>
 *   - BoundaryScheme::temporal_id_tag
 *   - Tags::Next<BoundaryScheme::temporal_id_tag>
 *   - Interface<Tags::Mesh<volume_dim - 1>>
 *   - Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>
 *   - Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: Tags::Mortars<BoundaryScheme::mortar_data_tag, volume_dim>
 *
 * \see ReceiveDataForFluxes
 */
template <typename BoundaryScheme>
struct SendDataForFluxes {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using fluxes_inbox_tag = dg::FluxesInboxTag<BoundaryScheme>;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const auto& all_mortar_data = get<all_mortar_data_tag>(box);
    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<temporal_id_tag>(box);
    const auto& mortar_meshes =
        db::get<Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>>(box);

    // Iterate over neighbors
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_and_neighbors.second;
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      for (const auto& neighbor : neighbors_in_direction) {
        const auto mortar_id = std::make_pair(direction, neighbor);

        // Make a copy of the local boundary data on the mortar to send to the
        // neighbor
        auto remote_boundary_data_on_mortar =
            all_mortar_data.at(mortar_id).local_data(temporal_id);

        // Reorient the data to the neighbor orientation if necessary
        if (not orientation.is_aligned()) {
          remote_boundary_data_on_mortar.orient_on_slice(
              mortar_meshes.at(mortar_id).extents(), dimension, orientation);
        }

        // Send remote data to neighbor
        Parallel::receive_data<fluxes_inbox_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                std::make_pair(temporal_id,
                               std::move(remote_boundary_data_on_mortar))));
      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
