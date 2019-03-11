// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines actions SendDataForFluxes and ReceiveDataForFluxes

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace dg {

/// The inbox tag for flux communication.
template <typename FluxLiftingScheme>
struct FluxesTag {
  static constexpr size_t volume_dim = FluxLiftingScheme::volume_dim;
  using temporal_id =
      db::item_type<typename FluxLiftingScheme::temporal_id_tag>;
  using type = std::map<
      temporal_id,
      FixedHashMap<
          maximum_number_of_neighbors(volume_dim),
          std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
          std::pair<temporal_id, typename FluxLiftingScheme::RemoteData>,
          boost::hash<
              std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>>;
};

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Receive boundary data needed for fluxes from neighbors.
///
/// Uses:
/// - DataBox:
///   - Metavariables::temporal_id
///   - Tags::Next<Metavariables::temporal_id>
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::Mortars<Tags::Next<Metavariables::temporal_id>, volume_dim>
///   - Tags::VariablesBoundaryData
///
/// \see SendDataForFluxes
template <typename FluxLiftingScheme>
struct ReceiveDataForFluxes {
 private:
  static constexpr size_t volume_dim = FluxLiftingScheme::volume_dim;
  using temporal_id_tag = typename FluxLiftingScheme::temporal_id_tag;
  using fluxes_tag = dg::FluxesTag<FluxLiftingScheme>;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename FluxLiftingScheme::mortar_data_tag, volume_dim>;

 public:
  using inbox_tags = tmpl::list<fluxes_tag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using neighbor_temporal_id_tag =
        Tags::Mortars<Tags::Next<temporal_id_tag>, volume_dim>;
    db::mutate<all_mortar_data_tag, neighbor_temporal_id_tag>(
        make_not_null(&box),
        [&inboxes](const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                       mortar_data,
                   const gsl::not_null<db::item_type<neighbor_temporal_id_tag>*>
                       neighbor_next_temporal_ids,
                   const db::item_type<Tags::Next<temporal_id_tag>>&
                       local_next_temporal_id) noexcept {
          auto& inbox = tuples::get<fluxes_tag>(inboxes);
          for (auto received_data = inbox.begin();
               received_data != inbox.end() and
               received_data->first < local_next_temporal_id;
               received_data = inbox.erase(received_data)) {
            const auto& receive_temporal_id = received_data->first;
            for (auto& received_mortar_data : received_data->second) {
              const auto mortar_id = received_mortar_data.first;
              ASSERT(neighbor_next_temporal_ids->at(mortar_id) ==
                         receive_temporal_id,
                     "Expected data at "
                         << neighbor_next_temporal_ids->at(mortar_id)
                         << " but received at " << receive_temporal_id);
              neighbor_next_temporal_ids->at(mortar_id) =
                  received_mortar_data.second.first;
              mortar_data->at(mortar_id).remote_insert(
                  receive_temporal_id,
                  std::move(received_mortar_data.second.second));
            }
          }

          // The apparently pointless lambda wrapping this check
          // prevents gcc-7.3.0 from segfaulting.
          ASSERT(([
                   &neighbor_next_temporal_ids, &local_next_temporal_id
                 ]() noexcept {
                   return std::all_of(
                       neighbor_next_temporal_ids->begin(),
                       neighbor_next_temporal_ids->end(),
                       [&local_next_temporal_id](const auto& next) noexcept {
                         return next.first.second ==
                                    ElementId<
                                        volume_dim>::external_boundary_id() or
                                next.second >= local_next_temporal_id;
                       });
                 }()),
                 "apply called before all data received");
          ASSERT(
              inbox.empty() or (inbox.size() == 1 and
                                inbox.begin()->first == local_next_temporal_id),
              "Shouldn't have received data that depended upon the step being "
              "taken: Received data at "
                  << inbox.begin()->first << " while stepping to "
                  << local_next_temporal_id);
        },
        db::get<Tags::Next<temporal_id_tag>>(box));

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox = tuples::get<fluxes_tag>(inboxes);
    const auto& local_next_temporal_id =
        db::get<Tags::Next<temporal_id_tag>>(box);
    const auto& mortars_next_temporal_id =
        db::get<Tags::Mortars<Tags::Next<temporal_id_tag>, volume_dim>>(box);
    for (const auto& mortar_id_next_temporal_id : mortars_next_temporal_id) {
      const auto& mortar_id = mortar_id_next_temporal_id.first;
      // If on an external boundary
      if (mortar_id.second == ElementId<volume_dim>::external_boundary_id()) {
        continue;
      }
      auto next_temporal_id = mortar_id_next_temporal_id.second;
      while (next_temporal_id < local_next_temporal_id) {
        const auto temporal_received = inbox.find(next_temporal_id);
        if (temporal_received == inbox.end()) {
          return false;
        }
        const auto mortar_received = temporal_received->second.find(mortar_id);
        if (mortar_received == temporal_received->second.end()) {
          return false;
        }
        next_temporal_id = mortar_received->second.first;
      }
    }
    return true;
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Send local boundary data needed for fluxes to neighbors.
///
/// With:
/// - `Interface<Tag> =
///   Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>`
///
/// Uses:
/// - ConstGlobalCache: Metavariables::normal_dot_numerical_flux
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - Interface<Tags listed in
///               Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - Interface<Tags::Mesh<volume_dim - 1>>
///   - Interface<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - Metavariables::temporal_id
///   - Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>
///   - Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>
///   - Tags::Next<Metavariables::temporal_id>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::VariablesBoundaryData
///
/// \see ReceiveDataForFluxes
template <typename FluxLiftingScheme>
struct SendDataForFluxes {
 private:
  static constexpr size_t volume_dim = FluxLiftingScheme::volume_dim;
  using temporal_id_tag = typename FluxLiftingScheme::temporal_id_tag;
  using fluxes_tag = dg::FluxesTag<FluxLiftingScheme>;

  template <typename F, typename DataBoxType, typename... ArgsTags,
            typename... ExtraArgs>
  static auto compute_packaged_face_data(
      const DataBoxType& box, const Direction<volume_dim>& direction,
      tmpl::list<ArgsTags...> /*meta*/,
      const ExtraArgs&... extra_args) noexcept {
    return F::apply(
        get<::Tags::Interface<::Tags::InternalDirections<volume_dim>,
                              ArgsTags>>(box)
            .at(direction)...,
        extra_args...);
  }

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

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<temporal_id_tag>(box);
    const auto& next_temporal_id = db::get<Tags::Next<temporal_id_tag>>(box);
    const auto& mortar_meshes =
        db::get<Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>>(box);
    const auto& mortar_sizes =
        db::get<Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>>(
            box);
    const auto& face_meshes =
        db::get<Tags::Interface<Tags::InternalDirections<volume_dim>,
                                Tags::Mesh<volume_dim - 1>>>(box);

    // Iterate over neighbors
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_and_neighbors.second;
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      // HACK until we can retrieve cache tags from the DataBox in compute items
      const auto remote_face_data = compute_packaged_face_data<
          typename FluxLiftingScheme::package_remote_data>(
          box, direction,
          typename FluxLiftingScheme::package_remote_data::argument_tags{},
          get<typename Metavariables::normal_dot_numerical_flux>(cache));
      const auto local_face_data = compute_packaged_face_data<
          typename FluxLiftingScheme::package_local_data>(
          box, direction,
          typename FluxLiftingScheme::package_local_data::argument_tags{},
          remote_face_data);

      for (const auto& neighbor : neighbors_in_direction) {
        const auto mortar_id = std::make_pair(direction, neighbor);

        // Project the packaged face data to this mortar
        auto remote_mortar_data = FluxLiftingScheme::project_to_mortar(
            remote_face_data, face_meshes.at(direction),
            mortar_meshes.at(mortar_id), mortar_sizes.at(mortar_id));
        auto local_mortar_data = FluxLiftingScheme::project_to_mortar(
            local_face_data, face_meshes.at(direction),
            mortar_meshes.at(mortar_id), mortar_sizes.at(mortar_id));

        // Reorient the variables to the neighbor orientation
        if (not orientation.is_aligned()) {
          remote_mortar_data = orient_variables_on_slice(
              remote_mortar_data, mortar_meshes.at(mortar_id).extents(),
              dimension, orientation);
        }

        // Send remote mortar data to neighbor
        Parallel::receive_data<fluxes_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                std::make_pair(next_temporal_id,
                               std::move(remote_mortar_data))));

        // Store local mortar data in DataBox
        using all_mortar_data_tag =
            ::Tags::Mortars<typename FluxLiftingScheme::mortar_data_tag,
                            volume_dim>;
        db::mutate<all_mortar_data_tag>(
            make_not_null(&box),
            [&mortar_id, &temporal_id, &local_mortar_data ](
                const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                    all_mortar_data) noexcept {
              all_mortar_data->at(mortar_id).local_insert(
                  temporal_id, std::move(local_mortar_data));
            });
      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
