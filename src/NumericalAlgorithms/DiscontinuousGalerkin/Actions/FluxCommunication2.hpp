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
  using mortars_temporal_id_tag =
      Tags::Mortars<Tags::Next<temporal_id_tag>, volume_dim>;

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
    db::mutate<all_mortar_data_tag, mortars_temporal_id_tag>(
        make_not_null(&box),
        [&inboxes](const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                       all_mortar_data,
                   const gsl::not_null<db::item_type<mortars_temporal_id_tag>*>
                       mortars_temporal_id,
                   const db::item_type<Tags::Next<temporal_id_tag>>&
                       local_next_temporal_id) noexcept {
          auto& inbox = tuples::get<fluxes_tag>(inboxes);
          // Iterate over all temporal ids that we have received data for,
          // up to this element's "next" temporal id
          for (auto received_data_all_mortars = inbox.begin();
               received_data_all_mortars != inbox.end() and
               received_data_all_mortars->first < local_next_temporal_id;
               received_data_all_mortars =
                   inbox.erase(received_data_all_mortars)) {
            const auto& received_temporal_id = received_data_all_mortars->first;
            // Iterate over all mortars that we have received data for at this
            // temporal id
            for (auto& received_data : received_data_all_mortars->second) {
              const auto mortar_id = received_data.first;
              // Increment the temporal id on the mortar to match the received
              // data, making sure that we have received data from all
              // intermediate steps
              ASSERT(mortars_temporal_id->at(mortar_id) == received_temporal_id,
                     "Expected data on mortar at temporal id "
                         << mortars_temporal_id->at(mortar_id)
                         << " but received data at " << received_temporal_id);
              mortars_temporal_id->at(mortar_id) = received_data.second.first;
              // Move the received data from the inbox into the mortar data
              all_mortar_data->at(mortar_id).remote_insert(
                  received_temporal_id, std::move(received_data.second.second));
            }
          }

          // At this point we are done with receiving data for this step. We
          // perform some sanity checks now.

          // Make sure we have received all data up to this element's "next"
          // temporal id.
          // The apparently pointless lambda wrapping this check
          // prevents gcc-7.3.0 from segfaulting.
          ASSERT(([&mortars_temporal_id, &local_next_temporal_id ]() noexcept {
                   return std::all_of(
                       mortars_temporal_id->begin(), mortars_temporal_id->end(),
                       [&local_next_temporal_id](
                           const auto& mortar_id_and_temporal_id) noexcept {
                         return mortar_id_and_temporal_id.first.second ==
                                    ElementId<
                                        volume_dim>::external_boundary_id() or
                                mortar_id_and_temporal_id.second >=
                                    local_next_temporal_id;
                       });
                 }()),
                 "Tried to complete flux communication step, but not all data "
                 "for this step has been received.");
          // Make sure we haven't received any data that we should not have been
          // able to get yet.
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
    const auto& mortars_temporal_id =
        db::get<Tags::Mortars<Tags::Next<temporal_id_tag>, volume_dim>>(box);
    for (const auto& mortar_id_and_temporal_id : mortars_temporal_id) {
      const auto& mortar_id = mortar_id_and_temporal_id.first;
      // Skip external boundaries
      if (mortar_id.second == ElementId<volume_dim>::external_boundary_id()) {
        continue;
      }
      // We are ready once data has been received for all temporal ids between
      // the current mortar temporal id and (excluding) the "next" temporal id
      // (We should probably remove the `Next` prefix from the temporal ids on
      // the mortar, since its meaning is not clear in this context)
      auto check_temporal_id = mortar_id_and_temporal_id.second;
      while (check_temporal_id < local_next_temporal_id) {
        const auto received_data_all_mortars = inbox.find(check_temporal_id);
        if (received_data_all_mortars == inbox.end()) {
          // No data has been received at this temporal id at all yet, so we're
          // not ready
          return false;
        }
        const auto received_data =
            received_data_all_mortars->second.find(mortar_id);
        if (received_data == received_data_all_mortars->second.end()) {
          // No data has been received at this temporal id for this mortar, so
          // we're not ready
          return false;
        }
        // Data has been received at this temporal id for this mortar, so check
        // the next intermediate temporal id until we have checked all temporal
        // ids within this element's temporal step
        check_temporal_id = received_data->second.first;
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
          get<typename FluxLiftingScheme::numerical_flux_computer_tag>(cache));
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
