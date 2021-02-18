// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Requires.hpp"
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

namespace elliptic::dg::Actions {

/*!
 * \brief Initialize the geometry for the DG subdomain operator
 *
 * Initializes tags that define the geometry of overlap regions with neighboring
 * elements. The data needs to be updated if the geometry of neighboring
 * elements changes.
 *
 * Note that the geometry depends on the system and on the choice of background
 * through the normalization of face normals, which involves a metric.
 */
template <typename System, typename BackgroundTag, typename OptionsGroup>
struct InitializeSubdomain {
 private:
  static constexpr size_t Dim = System::volume_dim;
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

  using geometry_tags = db::wrap_tags_in<
      overlaps_tag,
      tmpl::list<
          domain::Tags::Mesh<Dim>,
          elliptic::dg::subdomain_operator::Tags::ExtrudingExtent,
          domain::Tags::Element<Dim>, domain::Tags::ElementMap<Dim>,
          domain::Tags::Coordinates<Dim, Frame::Inertial>,
          domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<Dim>,
              domain::Tags::Coordinates<Dim, Frame::Inertial>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<Dim>,
              domain::Tags::Coordinates<Dim, Frame::Inertial>>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<Dim>,
              ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<Dim>,
              ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<Dim>,
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<Dim>,
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
          ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              domain::Tags::Mesh<Dim>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>,
              Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              domain::Tags::Mesh<Dim - 1>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              ::Tags::MortarSize<Dim - 1>, Dim>>>;

  // Possible optimization: Only include the background fields that are strictly
  // necessary for the DG operator, i.e. the background fields in the
  // System::fluxes_computer::argument_tags on internal faces, and possibly
  // additional background fields for boundary conditions.
  using background_tags = db::wrap_tags_in<
      overlaps_tag,
      tmpl::list<::Tags::Variables<typename System::background_fields>,
                 domain::Tags::Interface<
                     domain::Tags::InternalDirections<Dim>,
                     ::Tags::Variables<typename System::background_fields>>,
                 domain::Tags::Interface<
                     domain::Tags::BoundaryDirectionsInterior<Dim>,
                     ::Tags::Variables<typename System::background_fields>>>>;

 public:
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>>;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>;
  using simple_tags = tmpl::append<
      geometry_tags,
      tmpl::conditional_t<
          std::is_same_v<typename System::background_fields, tmpl::list<>>,
          tmpl::list<>, background_tags>>;
  using compute_tags = tmpl::list<>;

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::all<initialization_tags,
                         tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                    tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& initial_extents =
        db::get<domain::Tags::InitialExtents<Dim>>(box);
    const auto& initial_refinement =
        db::get<domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& max_overlap =
        get<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>(box);
    const auto& background = db::get<BackgroundTag>(box);

    overlaps<Mesh<Dim>> overlap_meshes{};
    overlaps<size_t> overlap_extents{};
    overlaps<Element<Dim>> overlap_elements{};
    overlaps<ElementMap<Dim, Frame::Inertial>> overlap_element_maps{};
    overlaps<tnsr::I<DataVector, Dim>> overlap_inertial_coords{};
    overlaps<InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>>
        overlap_inv_jacobians{};
    overlaps<Variables<typename System::background_fields>>
        overlap_background_fields{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>>
        overlap_face_inertial_coords_internal{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>>
        overlap_face_inertial_coords_external{};
    overlaps<std::unordered_map<Direction<Dim>,
                                Variables<typename System::background_fields>>>
        overlap_face_background_fields_internal{};
    overlaps<std::unordered_map<Direction<Dim>,
                                Variables<typename System::background_fields>>>
        overlap_face_background_fields_external{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>>
        overlap_face_normals_internal{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>>
        overlap_face_normals_external{};
    overlaps<std::unordered_map<Direction<Dim>, Scalar<DataVector>>>
        overlap_face_normal_magnitudes_internal{};
    overlaps<std::unordered_map<Direction<Dim>, Scalar<DataVector>>>
        overlap_face_normal_magnitudes_external{};
    overlaps<::dg::MortarMap<Dim, Mesh<Dim - 1>>> overlap_mortar_meshes{};
    overlaps<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>
        overlap_mortar_sizes{};
    overlaps<::dg::MortarMap<Dim, Mesh<Dim>>> overlap_neighbor_meshes{};
    overlaps<::dg::MortarMap<Dim, Scalar<DataVector>>>
        overlap_neighbor_face_normal_magnitudes{};
    overlaps<::dg::MortarMap<Dim, Mesh<Dim - 1>>>
        overlap_neighbor_mortar_meshes{};
    overlaps<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>
        overlap_neighbor_mortar_sizes{};

    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto& direction_from_neighbor = orientation(direction.opposite());
      const auto& dimension_in_neighbor = direction_from_neighbor.dimension();
      for (const auto& neighbor_id : neighbors) {
        const auto overlap_id = std::make_pair(direction, neighbor_id);
        // Mesh
        overlap_meshes.emplace(overlap_id,
                               domain::Initialization::create_initial_mesh(
                                   initial_extents, neighbor_id,
                                   Spectral::Quadrature::GaussLobatto));
        const auto& neighbor_mesh = overlap_meshes.at(overlap_id);
        // Overlap extents
        overlap_extents.emplace(
            overlap_id,
            LinearSolver::Schwarz::overlap_extent(
                neighbor_mesh.extents(dimension_in_neighbor), max_overlap));
        // Element
        const auto& neighbor_block = domain.blocks()[neighbor_id.block_id()];
        overlap_elements.emplace(
            overlap_id, domain::Initialization::create_initial_element(
                            neighbor_id, neighbor_block, initial_refinement));
        const auto& neighbor = overlap_elements.at(overlap_id);
        // Element map
        overlap_element_maps.emplace(
            overlap_id,
            ElementMap<Dim, Frame::Inertial>{
                neighbor_id, neighbor_block.stationary_map().get_clone()});
        const auto& neighbor_element_map = overlap_element_maps.at(overlap_id);
        // Inertial coords
        const auto neighbor_logical_coords = logical_coordinates(neighbor_mesh);
        const auto& neighbor_inertial_coords =
            overlap_inertial_coords
                .emplace(overlap_id,
                         neighbor_element_map(neighbor_logical_coords))
                .first->second;
        // Jacobian
        overlap_inv_jacobians.emplace(
            overlap_id,
            neighbor_element_map.inv_jacobian(neighbor_logical_coords));
        // Background fields
        if constexpr (not std::is_same_v<typename System::background_fields,
                                         tmpl::list<>>) {
          overlap_background_fields.emplace(
              overlap_id, variables_from_tagged_tuple(background.variables(
                              neighbor_inertial_coords,
                              typename System::background_fields{})));
        }
        // Faces and mortars, internal and external
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>
            neighbor_face_inertial_coords_internal{};
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>
            neighbor_face_inertial_coords_external{};
        std::unordered_map<Direction<Dim>,
                           Variables<typename System::background_fields>>
            neighbor_face_background_fields_internal{};
        std::unordered_map<Direction<Dim>,
                           Variables<typename System::background_fields>>
            neighbor_face_background_fields_external{};
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>
            neighbor_face_normals_internal{};
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>
            neighbor_face_normals_external{};
        std::unordered_map<Direction<Dim>, Scalar<DataVector>>
            neighbor_face_normal_magnitudes_internal{};
        std::unordered_map<Direction<Dim>, Scalar<DataVector>>
            neighbor_face_normal_magnitudes_external{};
        const auto setup_face = [&neighbor_face_background_fields_internal,
                                 &neighbor_face_background_fields_external,
                                 &neighbor_face_inertial_coords_internal,
                                 &neighbor_face_inertial_coords_external,
                                 &neighbor_face_normals_internal,
                                 &neighbor_face_normals_external,
                                 &neighbor_face_normal_magnitudes_internal,
                                 &neighbor_face_normal_magnitudes_external,
                                 &neighbor_mesh, &neighbor_element_map,
                                 &overlap_background_fields, &overlap_id](
                                    const Direction<Dim>& local_direction,
                                    const bool is_external) {
          auto& neighbor_face_inertial_coords =
              is_external ? neighbor_face_inertial_coords_external
                          : neighbor_face_inertial_coords_internal;
          auto& neighbor_face_background_fields =
              is_external ? neighbor_face_background_fields_external
                          : neighbor_face_background_fields_internal;
          auto& neighbor_face_normals = is_external
                                            ? neighbor_face_normals_external
                                            : neighbor_face_normals_internal;
          auto& neighbor_face_normal_magnitudes =
              is_external ? neighbor_face_normal_magnitudes_external
                          : neighbor_face_normal_magnitudes_internal;
          const auto neighbor_face_mesh =
              neighbor_mesh.slice_away(local_direction.dimension());
          neighbor_face_inertial_coords.emplace(
              local_direction,
              neighbor_element_map(interface_logical_coordinates(
                  neighbor_face_mesh, local_direction)));
          if constexpr (not std::is_same_v<typename System::background_fields,
                                           tmpl::list<>>) {
            // Slicing the background fields to the face instead of evaluating
            // them on the face coords to avoid re-computing them, and because
            // this is also what the DG operator currently does. The result is
            // equivalent on Gauss-Lobatto grids, but needs adjusting when
            // adding support for Gauss grids.
            neighbor_face_background_fields.emplace(
                local_direction,
                data_on_slice(overlap_background_fields.at(overlap_id),
                              neighbor_mesh.extents(),
                              local_direction.dimension(),
                              index_to_slice_at(neighbor_mesh.extents(),
                                                local_direction)));
          } else {
            (void)overlap_background_fields;
            (void)overlap_id;
          }
          auto neighbor_face_normal = unnormalized_face_normal(
              neighbor_face_mesh, neighbor_element_map, local_direction);
          Scalar<DataVector> neighbor_normal_magnitude{
              neighbor_face_mesh.number_of_grid_points()};
          if constexpr (std::is_same_v<typename System::inv_metric_tag, void>) {
            magnitude(make_not_null(&neighbor_normal_magnitude),
                      neighbor_face_normal);
          } else {
            magnitude(make_not_null(&neighbor_normal_magnitude),
                      neighbor_face_normal,
                      get<typename System::inv_metric_tag>(
                          neighbor_face_background_fields.at(local_direction)));
          }
          for (size_t d = 0; d < Dim; d++) {
            neighbor_face_normal.get(d) /= get(neighbor_normal_magnitude);
          }
          neighbor_face_normals.emplace(local_direction,
                                        std::move(neighbor_face_normal));
          neighbor_face_normal_magnitudes.emplace(
              local_direction, std::move(neighbor_normal_magnitude));
        };
        ::dg::MortarMap<Dim, Mesh<Dim - 1>> neighbor_mortar_meshes{};
        ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>> neighbor_mortar_sizes{};
        for (const auto& [neighbor_direction, neighbor_neighbors] :
             neighbor.neighbors()) {
          setup_face(neighbor_direction, false);
          const auto neighbor_dimension = neighbor_direction.dimension();
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
                                      Spectral::Quadrature::GaussLobatto,
                                      neighbor_neighbors.orientation())
                                      .slice_away(neighbor_dimension)));
            neighbor_mortar_sizes.emplace(
                neighbor_mortar_id,
                ::dg::mortar_size(neighbor_id, neighbor_neighbor_id,
                                  neighbor_dimension,
                                  neighbor_neighbors.orientation()));
          }
        }
        for (const auto& neighbor_direction : neighbor.external_boundaries()) {
          setup_face(neighbor_direction, true);
          const auto neighbor_mortar_id = std::make_pair(
              neighbor_direction, ElementId<Dim>::external_boundary_id());
          neighbor_mortar_meshes.emplace(
              neighbor_mortar_id,
              neighbor_mesh.slice_away(neighbor_direction.dimension()));
          neighbor_mortar_sizes.emplace(
              neighbor_mortar_id,
              make_array<Dim - 1>(Spectral::MortarSize::Full));
        }
        overlap_face_inertial_coords_internal.emplace(
            overlap_id, std::move(neighbor_face_inertial_coords_internal));
        overlap_face_inertial_coords_external.emplace(
            overlap_id, std::move(neighbor_face_inertial_coords_external));
        overlap_face_background_fields_internal.emplace(
            overlap_id, std::move(neighbor_face_background_fields_internal));
        overlap_face_background_fields_external.emplace(
            overlap_id, std::move(neighbor_face_background_fields_external));
        overlap_face_normals_internal.emplace(
            overlap_id, std::move(neighbor_face_normals_internal));
        overlap_face_normals_external.emplace(
            overlap_id, std::move(neighbor_face_normals_external));
        overlap_face_normal_magnitudes_internal.emplace(
            overlap_id, std::move(neighbor_face_normal_magnitudes_internal));
        overlap_face_normal_magnitudes_external.emplace(
            overlap_id, std::move(neighbor_face_normal_magnitudes_external));
        overlap_mortar_meshes.emplace(overlap_id,
                                      std::move(neighbor_mortar_meshes));
        overlap_mortar_sizes.emplace(overlap_id,
                                     std::move(neighbor_mortar_sizes));

        // Neighbor's neighbors
        ::dg::MortarMap<Dim, Mesh<Dim>> neighbors_neighbor_meshes{};
        ::dg::MortarMap<Dim, Scalar<DataVector>>
            neighbors_neighbor_face_normal_magnitudes{};
        ::dg::MortarMap<Dim, Mesh<Dim - 1>> neighbors_neighbor_mortar_meshes{};
        ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>
            neighbors_neighbor_mortar_sizes{};
        for (const auto& [neighbor_direction, neighbor_neighbors] :
             neighbor.neighbors()) {
          const auto& neighbors_neighbor_orientation =
              neighbor_neighbors.orientation();
          const auto direction_from_neighbors_neighbor =
              neighbors_neighbor_orientation(neighbor_direction.opposite());
          const auto reoriented_neighbor_face_mesh =
              neighbors_neighbor_orientation(neighbor_mesh)
                  .slice_away(direction_from_neighbors_neighbor.dimension());
          for (const auto& neighbors_neighbor_id : neighbor_neighbors) {
            const auto neighbors_neighbor_mortar_id =
                std::make_pair(neighbor_direction, neighbors_neighbor_id);
            neighbors_neighbor_meshes.emplace(
                neighbors_neighbor_mortar_id,
                domain::Initialization::create_initial_mesh(
                    initial_extents, neighbors_neighbor_id,
                    Spectral::Quadrature::GaussLobatto));
            const auto& neighbors_neighbor_mesh =
                neighbors_neighbor_meshes.at(neighbors_neighbor_mortar_id);
            const auto neighbors_neighbor_face_mesh =
                neighbors_neighbor_mesh.slice_away(
                    direction_from_neighbors_neighbor.dimension());
            const auto& neighbors_neighbor_block =
                domain.blocks()[neighbors_neighbor_id.block_id()];
            ElementMap<Dim, Frame::Inertial> neighbors_neighbor_element_map{
                neighbors_neighbor_id,
                neighbors_neighbor_block.stationary_map().get_clone()};
            const auto neighbors_neighbor_face_normal =
                unnormalized_face_normal(neighbors_neighbor_face_mesh,
                                         neighbors_neighbor_element_map,
                                         direction_from_neighbors_neighbor);
            Scalar<DataVector> neighbors_neighbor_face_normal_magnitude{
                neighbors_neighbor_face_mesh.number_of_grid_points()};
            if constexpr (std::is_same_v<typename System::inv_metric_tag,
                                         void>) {
              magnitude(
                  make_not_null(&neighbors_neighbor_face_normal_magnitude),
                  neighbors_neighbor_face_normal);
            } else {
              const auto neighbors_neighbor_face_inertial_coords =
                  neighbors_neighbor_element_map(interface_logical_coordinates(
                      neighbors_neighbor_face_mesh,
                      direction_from_neighbors_neighbor));
              magnitude(
                  make_not_null(&neighbors_neighbor_face_normal_magnitude),
                  neighbors_neighbor_face_normal,
                  get<typename System::inv_metric_tag>(background.variables(
                      neighbors_neighbor_face_inertial_coords,
                      tmpl::list<typename System::inv_metric_tag>{})));
            }
            neighbors_neighbor_face_normal_magnitudes.emplace(
                neighbors_neighbor_mortar_id,
                std::move(neighbors_neighbor_face_normal_magnitude));
            neighbors_neighbor_mortar_meshes.emplace(
                neighbors_neighbor_mortar_id,
                ::dg::mortar_mesh(reoriented_neighbor_face_mesh,
                                  neighbors_neighbor_face_mesh));
            neighbors_neighbor_mortar_sizes.emplace(
                neighbors_neighbor_mortar_id,
                ::dg::mortar_size(
                    neighbors_neighbor_id, neighbor_id,
                    direction_from_neighbors_neighbor.dimension(),
                    neighbors_neighbor_orientation.inverse_map()));
          }
        }
        overlap_neighbor_meshes.emplace(overlap_id,
                                        std::move(neighbors_neighbor_meshes));
        overlap_neighbor_face_normal_magnitudes.emplace(
            overlap_id, std::move(neighbors_neighbor_face_normal_magnitudes));
        overlap_neighbor_mortar_meshes.emplace(
            overlap_id, std::move(neighbors_neighbor_mortar_meshes));
        overlap_neighbor_mortar_sizes.emplace(
            overlap_id, std::move(neighbors_neighbor_mortar_sizes));
      }  // neighbors in direction
    }    // directions

    ::Initialization::mutate_assign<geometry_tags>(
        make_not_null(&box), std::move(overlap_meshes),
        std::move(overlap_extents), std::move(overlap_elements),
        std::move(overlap_element_maps), std::move(overlap_inertial_coords),
        std::move(overlap_inv_jacobians),
        std::move(overlap_face_inertial_coords_internal),
        std::move(overlap_face_inertial_coords_external),
        std::move(overlap_face_normals_internal),
        std::move(overlap_face_normals_external),
        std::move(overlap_face_normal_magnitudes_internal),
        std::move(overlap_face_normal_magnitudes_external),
        std::move(overlap_mortar_meshes), std::move(overlap_mortar_sizes),
        std::move(overlap_neighbor_meshes),
        std::move(overlap_neighbor_face_normal_magnitudes),
        std::move(overlap_neighbor_mortar_meshes),
        std::move(overlap_neighbor_mortar_sizes));
    if constexpr (not std::is_same_v<typename System::background_fields,
                                     tmpl::list<>>) {
      ::Initialization::mutate_assign<background_tags>(
          make_not_null(&box), std::move(overlap_background_fields),
          std::move(overlap_face_background_fields_internal),
          std::move(overlap_face_background_fields_external));
    }
    return {std::move(box)};
  }

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::all<initialization_tags,
                             tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                        tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }
};

}  // namespace elliptic::dg::Actions
