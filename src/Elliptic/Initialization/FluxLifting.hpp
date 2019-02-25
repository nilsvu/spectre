// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Evolution/Initialization/Helpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Initializes DataBox tags related to discontinuous Galerkin fluxes
 *
 * With:
 * - `flux_comm_types` = `dg::FluxCommunicationTypes<Metavariables>`
 * - `mortar_data_tag` = `flux_comm_types::simple_mortar_data_tag`
 * - `interface<Tag>` =
 * `Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>`
 * - `boundary<Tag>` =
 * `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 * - `mortar<Tag>` = `Tags::Mortars<Tag, volume_dim>`
 *
 * Uses:
 * - Metavariables:
 *   - `temporal_id`
 *   - Items required by `flux_comm_types`
 * - System:
 *   - `volume_dim`
 * - DataBox:
 *   - `Tags::Element<volume_dim>`
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::InternalDirections<volume_dim>`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `interface<Tags::Mesh<volume_dim - 1>>`
 *   - `boundary<Tags::Mesh<volume_dim - 1>>`
 *
 * DataBox:
 * - Adds:
 *   - `temporal_id`
 *   - `mortar_data_tag`
 *   - `mortar<Tags::Next<temporal_id>>`
 *   - `mortar<Tags::Mesh<volume_dim - 1>>`
 *   - `mortar<Tags::MortarSize<volume_dim - 1>>`
 *   - `interface<flux_comm_types::normal_dot_fluxes_tag>`
 *   - `boundary<flux_comm_types::normal_dot_fluxes_tag>`
 */
template <typename FluxLiftingScheme>
struct FluxLifting {
  static constexpr size_t volume_dim = FluxLiftingScheme::volume_dim;
  using temporal_id_tag = typename FluxLiftingScheme::temporal_id_tag;
  using next_temporal_id_tag =
      ::Tags::Next<typename FluxLiftingScheme::temporal_id_tag>;
  using mortar_data_tag =
      ::Tags::Mortars<typename FluxLiftingScheme::mortar_data_tag, volume_dim>;

  template <typename Tag>
  using interface_tag =
      ::Tags::Interface<::Tags::InternalDirections<volume_dim>, Tag>;
  template <typename Tag>
  using boundary_tag =
      ::Tags::Interface<::Tags::BoundaryDirectionsInterior<volume_dim>, Tag>;
  template <typename Tag>
  using mortar_tag = ::Tags::Mortars<Tag, volume_dim>;

  using simple_tags = db::AddSimpleTags<
      mortar_data_tag, mortar_tag<::Tags::Next<temporal_id_tag>>,
      interface_tag<typename FluxLiftingScheme::normal_dot_fluxes_tag>,
      boundary_tag<typename FluxLiftingScheme::normal_dot_fluxes_tag>>;

  using compute_tags = db::AddComputeTags<>;

  // This function is mostly copied from
  // `Evolution/Initialization/DiscontinuousGalerkin.hpp`.
  // It could be useful to move some of this functionality into compute items.
  template <typename TagsList>
  static auto add_mortar_data(db::DataBox<TagsList>&& box,
                              const std::vector<std::array<size_t, volume_dim>>&
                              /*initial_extents*/) noexcept {
    const auto& element = db::get<::Tags::Element<volume_dim>>(box);

    db::item_type<mortar_data_tag> mortar_data{};
    db::item_type<mortar_tag<::Tags::Next<temporal_id_tag>>>
        mortar_next_temporal_ids{};
    const auto& temporal_id = get<temporal_id_tag>(box);
    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const auto& neighbors = direction_neighbors.second;
      for (const auto& neighbor : neighbors) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        mortar_data[mortar_id];  // Default initialize data
        mortar_next_temporal_ids.insert({mortar_id, temporal_id});
      }
    }
    // Add mortars for external boundaries
    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());
      mortar_data[mortar_id];  // Default initialize data
      mortar_next_temporal_ids.insert({mortar_id, temporal_id});
    }

    return db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<mortar_data_tag,
                          mortar_tag<::Tags::Next<temporal_id_tag>>>>(
        std::move(box), std::move(mortar_data),
        std::move(mortar_next_temporal_ids));
  }

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box,
                         const std::vector<std::array<size_t, volume_dim>>&
                             initial_extents) noexcept {
    auto temporal_box = db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                                        db::AddComputeTags<>>(std::move(box));

    auto mortar_box = add_mortar_data(std::move(temporal_box), initial_extents);

    const auto& internal_directions =
        db::get<::Tags::InternalDirections<volume_dim>>(mortar_box);
    const auto& boundary_directions =
        db::get<::Tags::BoundaryDirectionsInterior<volume_dim>>(mortar_box);

    db::item_type<
        interface_tag<typename FluxLiftingScheme::normal_dot_fluxes_tag>>
        normal_dot_fluxes{};
    for (const auto& direction : internal_directions) {
      const auto& interface_num_points =
          db::get<interface_tag<::Tags::Mesh<volume_dim - 1>>>(mortar_box)
              .at(direction)
              .number_of_grid_points();
      normal_dot_fluxes[direction].initialize(interface_num_points, 0.);
    }
    db::item_type<
        boundary_tag<typename FluxLiftingScheme::normal_dot_fluxes_tag>>
        boundary_normal_dot_fluxes{};
    for (const auto& direction : boundary_directions) {
      const auto& interface_num_points =
          db::get<boundary_tag<::Tags::Mesh<volume_dim - 1>>>(mortar_box)
              .at(direction)
              .number_of_grid_points();
      boundary_normal_dot_fluxes[direction].initialize(interface_num_points,
                                                       0.);
    }

    return db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<
            interface_tag<typename FluxLiftingScheme::normal_dot_fluxes_tag>,
            boundary_tag<typename FluxLiftingScheme::normal_dot_fluxes_tag>>>(
        std::move(mortar_box), std::move(normal_dot_fluxes),
        std::move(boundary_normal_dot_fluxes));
  }

  // This function is called as a simple action for re-initialization.
  // When we do initialization in a phase-dependent action list, both functions
  // in this struct will be made into iterative actions.
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<mortar_tag<::Tags::Next<temporal_id_tag>>>(
        make_not_null(&box),
        [](const gsl::not_null<
               db::item_type<mortar_tag<::Tags::Next<temporal_id_tag>>>*>
               mortars_temporal_id,
           const db::item_type<temporal_id_tag>& temporal_id) noexcept {
          for (auto& mortar_id_and_temporal_id : *mortars_temporal_id) {
            mortar_id_and_temporal_id.second = temporal_id;
          }
        },
        get<temporal_id_tag>(box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Initialization
}  // namespace Elliptic
