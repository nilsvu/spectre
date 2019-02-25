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
template <typename Metavariables>
struct DiscontinuousGalerkin {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;
  template <typename Tag>
  using mortar_tag = ::Tags::Mortars<Tag, volume_dim>;

  using simple_tags =
      db::AddSimpleTags<mortar_tag<::Tags::Mesh<volume_dim - 1>>,
                        mortar_tag<::Tags::MortarSize<volume_dim - 1>>>;

  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box,
                         const std::vector<std::array<size_t, volume_dim>>&
                             initial_extents) noexcept {
    const auto& element = db::get<::Tags::Element<volume_dim>>(box);
    const auto& mesh = db::get<::Tags::Mesh<volume_dim>>(box);

    db::item_type<mortar_tag<::Tags::Mesh<volume_dim - 1>>> mortar_meshes{};
    db::item_type<mortar_tag<::Tags::MortarSize<volume_dim - 1>>>
        mortar_sizes{};
    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const auto& neighbors = direction_neighbors.second;
      for (const auto& neighbor : neighbors) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        mortar_meshes.emplace(
            mortar_id,
            ::dg::mortar_mesh(
                mesh.slice_away(direction.dimension()),
                ::Initialization::element_mesh(initial_extents, neighbor,
                                               neighbors.orientation())
                    .slice_away(direction.dimension())));
        mortar_sizes.emplace(
            mortar_id,
            ::dg::mortar_size(element.id(), neighbor, direction.dimension(),
                              neighbors.orientation()));
      }
    }
    // Add mortars for external boundaries
    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());
      mortar_meshes.emplace(mortar_id, mesh.slice_away(direction.dimension()));
      mortar_sizes.emplace(
          mortar_id, make_array<volume_dim - 1>(Spectral::MortarSize::Full));
    }

    return db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<mortar_tag<::Tags::Mesh<volume_dim - 1>>,
                          mortar_tag<::Tags::MortarSize<volume_dim - 1>>>>(
        std::move(box), std::move(mortar_meshes), std::move(mortar_sizes));
  }
};
}  // namespace Initialization
}  // namespace Elliptic
