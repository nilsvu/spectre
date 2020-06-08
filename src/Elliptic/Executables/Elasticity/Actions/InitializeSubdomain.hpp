// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/TMPL.hpp"

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

namespace SolveElasticity::Actions {

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
                 domain::Tags::Faces<
                     Dim, domain::Tags::Coordinates<Dim, Frame::Inertial>>>>;
  using compute_tags = tmpl::list<>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<DataBox&&> apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& overlap_meshes =
        db::get<overlaps_tag<domain::Tags::Mesh<Dim>>>(box);
    const auto& overlap_element_maps =
        db::get<overlaps_tag<domain::Tags::ElementMap<Dim>>>(box);
    const auto& overlap_elements =
        db::get<overlaps_tag<domain::Tags::Element<Dim>>>(box);

    overlaps<tnsr::I<DataVector, Dim>> overlap_inertial_coords{};
    overlaps<DirectionMap<Dim, tnsr::I<DataVector, Dim>>>
        overlap_face_inertial_coords{};

    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const auto& neighbor_id : neighbors) {
        const auto overlap_id = std::make_pair(direction, neighbor_id);
        const auto& neighbor_mesh = overlap_meshes.at(overlap_id);
        const auto& neighbor_element_map = overlap_element_maps.at(overlap_id);
        const auto& neighbor = overlap_elements.at(overlap_id);
        // Coords
        const auto neighbor_logical_coords = logical_coordinates(neighbor_mesh);
        overlap_inertial_coords.emplace(
            overlap_id, neighbor_element_map(neighbor_logical_coords));
        // Face coords
        DirectionMap<Dim, tnsr::I<DataVector, Dim>>
            neighbor_face_inertial_coords{};
        const auto setup_face =
            [&neighbor_mesh, &neighbor_element_map,
             &neighbor_face_inertial_coords](
                const Direction<Dim>& local_direction) {
              const auto neighbor_face_mesh =
                  neighbor_mesh.slice_away(local_direction.dimension());
              const auto neighbor_face_logical_coords =
                  interface_logical_coordinates(neighbor_face_mesh,
                                                local_direction);
              neighbor_face_inertial_coords.emplace(
                  local_direction,
                  neighbor_element_map(neighbor_face_logical_coords));
            };
        for (const auto& [neighbor_direction, neighbor_neighbors] :
             neighbor.neighbors()) {
          (void)neighbor_neighbors;
          setup_face(neighbor_direction);
        }
        for (const auto& neighbor_direction : neighbor.external_boundaries()) {
          setup_face(neighbor_direction);
        }
        overlap_face_inertial_coords.emplace(
            overlap_id, std::move(neighbor_face_inertial_coords));
      }  // neighbors in direction
    }    // directions

    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(overlap_inertial_coords),
        std::move(overlap_face_inertial_coords));
    return {std::move(box)};
  }
};

}  // namespace SolveElasticity::Actions
