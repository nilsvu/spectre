// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/CreateInitialMesh.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::dg::Actions {

/*!
 * \brief Initialize DataBox tags related to the DG subdomain operator
 *
 * Initializes tags on overlap regions with neighboring elements. The data needs
 * to be updated if the geometry of neighboring elements changes.
 */
// TODO: test this action to make sure it is consistent with other dg actions
// initializing these items on elements
// TODO: Are h-refined mortars weighted correctly?
// TODO: Keep in mind that the weighting operation should preserve symmetry of
// the linear operator
template <size_t Dim, typename OptionsGroup,
          typename BoundaryConditionsProviderTag>
struct InitializeSubdomain {
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>>;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>;

  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::all<initialization_tags,
                         tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                    tmpl::pin<DataBox>>>::value> = nullptr>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& /*element_id*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& initial_extents =
        db::get<domain::Tags::InitialExtents<Dim>>(box);
    const auto& initial_refinement =
        db::get<domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& max_overlap =
        get<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>(box);
    const auto& boundary_conditions_provider =
        db::get<BoundaryConditionsProviderTag>(box);

    overlaps<Mesh<Dim>> overlap_meshes{};
    overlaps<Index<Dim>> overlap_extents{};
    overlaps<ElementMap<Dim, Frame::Inertial>> overlap_element_maps{};
    overlaps<::dg::MortarMap<Dim, Mesh<Dim - 1>>> overlap_mortar_meshes{};
    overlaps<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>
        overlap_mortar_sizes{};
    overlaps<tnsr::I<DataVector, Dim, Frame::Inertial>>
        overlap_inertial_coords{};
    overlaps<std::unordered_map<Direction<Dim>, elliptic::BoundaryCondition>>
        overlap_boundary_conditions{};

    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const auto& neighbors = direction_and_neighbors.second;
      const auto& orientation = neighbors.orientation();
      const auto& direction_from_neighbor = orientation(direction.opposite());
      const auto& dimension_in_neighbor = direction_from_neighbor.dimension();
      for (const auto& neighbor_id : neighbors) {
        const auto overlap_id = std::make_pair(direction, neighbor_id);
        // Mesh
        overlap_meshes.emplace(overlap_id,
                               domain::Initialization::create_initial_mesh(
                                   initial_extents, neighbor_id));
        const auto& neighbor_mesh = overlap_meshes.at(overlap_id);
        // Overlap extents
        overlap_extents.emplace(
            overlap_id,
            LinearSolver::Schwarz::overlap_extents(
                neighbor_mesh.extents(), max_overlap, dimension_in_neighbor));
        // Element map
        const auto& neighbor_block = domain.blocks()[neighbor_id.block_id()];
        overlap_element_maps.emplace(
            overlap_id,
            ElementMap<Dim, Frame::Inertial>{
                neighbor_id, neighbor_block.stationary_map().get_clone()});
        // Mortars
        Element<Dim> neighbor = domain::Initialization::create_initial_element(
            neighbor_id, neighbor_block, initial_refinement);
        ::dg::MortarMap<Dim, Mesh<Dim - 1>> neighbor_mortar_meshes{};
        ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>> neighbor_mortar_sizes{};
        for (const auto& neighbor_direction_and_neighbors :
             neighbor.neighbors()) {
          const auto& neighbor_direction =
              neighbor_direction_and_neighbors.first;
          const auto neighbor_dimension = neighbor_direction.dimension();
          const auto& neighbor_neighbors =
              neighbor_direction_and_neighbors.second;
          const auto neighbor_face_mesh =
              neighbor_mesh.slice_away(neighbor_dimension);
          for (const auto& neighbor_neighbor_id : neighbor_neighbors) {
            const auto neighbor_mortar_id =
                std::make_pair(neighbor_direction, neighbor_neighbor_id);
            neighbor_mortar_meshes.emplace(
                neighbor_mortar_id,
                ::dg::mortar_mesh(neighbor_face_mesh,
                                  domain::Initialization::create_initial_mesh(
                                      initial_extents, neighbor_neighbor_id,
                                      neighbor_neighbors.orientation())
                                      .slice_away(neighbor_dimension)));
            neighbor_mortar_sizes.emplace(
                neighbor_mortar_id,
                ::dg::mortar_size(neighbor_id, neighbor_neighbor_id,
                                  neighbor_dimension,
                                  neighbor_neighbors.orientation()));
          }
        }
        std::unordered_map<Direction<Dim>, elliptic::BoundaryCondition>
            neighbor_boundary_conditions{};
        for (const auto& neighbor_direction : neighbor.external_boundaries()) {
          const auto neighbor_mortar_id = std::make_pair(
              neighbor_direction, ElementId<Dim>::external_boundary_id());
          neighbor_mortar_meshes.emplace(
              neighbor_mortar_id,
              neighbor_mesh.slice_away(neighbor_direction.dimension()));
          neighbor_mortar_sizes.emplace(
              neighbor_mortar_id,
              make_array<Dim - 1>(Spectral::MortarSize::Full));
          neighbor_boundary_conditions.emplace(
              neighbor_direction,
              boundary_conditions_provider.boundary_condition_type(
                  overlap_element_maps.at(overlap_id)(
                      interface_logical_coordinates(
                          neighbor_mortar_meshes.at(neighbor_mortar_id),
                          neighbor_direction)),
                  neighbor_direction));
        }
        overlap_mortar_meshes.emplace(overlap_id,
                                      std::move(neighbor_mortar_meshes));
        overlap_mortar_sizes.emplace(overlap_id,
                                     std::move(neighbor_mortar_sizes));
        overlap_boundary_conditions.emplace(
            overlap_id, std::move(neighbor_boundary_conditions));
        auto neighbor_inertial_coords = overlap_element_maps.at(overlap_id)(
            logical_coordinates(neighbor_mesh));
        overlap_inertial_coords.emplace(overlap_id,
                                        std::move(neighbor_inertial_coords));
      }
    }

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeSubdomain,
            db::AddSimpleTags<
                overlaps_tag<domain::Tags::Mesh<Dim>>,
                overlaps_tag<domain::Tags::Extents<Dim>>,
                overlaps_tag<domain::Tags::ElementMap<Dim>>,
                overlaps_tag<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
                overlaps_tag<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>,
                overlaps_tag<domain::Tags::Coordinates<Dim, Frame::Inertial>>,
                overlaps_tag<domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsExterior<Dim>,
                    elliptic::Tags::BoundaryCondition>>>>(
            std::move(box), std::move(overlap_meshes),
            std::move(overlap_extents), std::move(overlap_element_maps),
            std::move(overlap_mortar_meshes), std::move(overlap_mortar_sizes),
            std::move(overlap_inertial_coords),
            std::move(overlap_boundary_conditions)));
  }

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::all<initialization_tags,
                             tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                        tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }
};

}  // namespace elliptic::dg::Actions
