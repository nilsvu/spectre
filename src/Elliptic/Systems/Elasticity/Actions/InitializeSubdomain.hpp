// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Elasticity::Actions {

// Initialize data on overlaps needed for the elasticity system
template <size_t Dim, typename OptionsGroup>
struct InitializeSubdomain {
 private:
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

 public:
  using simple_tags = db::wrap_tags_in<
      overlaps_tag,
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 domain::Tags::Interface<
                     domain::Tags::InternalDirections<Dim>,
                     domain::Tags::Coordinates<Dim, Frame::Inertial>>,
                 domain::Tags::Interface<
                     domain::Tags::BoundaryDirectionsInterior<Dim>,
                     domain::Tags::Coordinates<Dim, Frame::Inertial>>>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& overlap_meshes =
        db::get<overlaps_tag<domain::Tags::Mesh<Dim>>>(box);
    const auto& overlap_elements =
        db::get<overlaps_tag<domain::Tags::Element<Dim>>>(box);
    const auto& overlap_element_maps =
        db::get<overlaps_tag<domain::Tags::ElementMap<Dim>>>(box);

    overlaps<tnsr::I<DataVector, Dim>> overlap_inertial_coords{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>>
        overlap_boundary_inertial_coords_internal{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>>
        overlap_boundary_inertial_coords_external{};

    for (const auto& [overlap_id, neighbor_element_map] :
         overlap_element_maps) {
      const auto& neighbor_mesh = overlap_meshes.at(overlap_id);
      const auto& neighbor = overlap_elements.at(overlap_id);

      // Coords on the overlapped neighbor
      overlap_inertial_coords.emplace(
          overlap_id, neighbor_element_map(logical_coordinates(neighbor_mesh)));
      const auto& neighbor_inertial_coords =
          overlap_inertial_coords.at(overlap_id);

      // Coords on the faces of the overlapped neighbor
      std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>
          neighbor_boundary_inertial_coords_internal{};
      std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>
          neighbor_boundary_inertial_coords_external{};
      const auto setup_face = [&neighbor_inertial_coords, &neighbor_mesh,
                               &neighbor_boundary_inertial_coords_internal,
                               &neighbor_boundary_inertial_coords_external](
                                  const Direction<Dim>& local_direction,
                                  const bool is_external) {
        auto& neighbor_boundary_inertial_coords =
            is_external ? neighbor_boundary_inertial_coords_external
                        : neighbor_boundary_inertial_coords_internal;
        neighbor_boundary_inertial_coords.emplace(
            local_direction,
            data_on_slice(
                neighbor_inertial_coords, neighbor_mesh.extents(),
                local_direction.dimension(),
                index_to_slice_at(neighbor_mesh.extents(), local_direction)));
      };
      for (const auto& direction_from_neighbor :
           neighbor.internal_boundaries()) {
        setup_face(direction_from_neighbor, false);
      }
      for (const auto& direction_from_neighbor :
           neighbor.external_boundaries()) {
        setup_face(direction_from_neighbor, true);
      }
      overlap_boundary_inertial_coords_internal.emplace(
          overlap_id, std::move(neighbor_boundary_inertial_coords_internal));
      overlap_boundary_inertial_coords_external.emplace(
          overlap_id, std::move(neighbor_boundary_inertial_coords_external));
    }

    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(overlap_inertial_coords),
        std::move(overlap_boundary_inertial_coords_internal),
        std::move(overlap_boundary_inertial_coords_external));
    return {std::move(box)};
  }
};

}  // namespace Elasticity::Actions
