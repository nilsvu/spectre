// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
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
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::dg::Actions {

namespace detail {
template <size_t Dim, typename OptionsGroup, typename DbTags, typename... Tags>
auto add_tags(db::DataBox<DbTags>&& box, tmpl::list<Tags...> /*meta*/) {
  return db::create_from<
      db::RemoveTags<>,
      db::AddSimpleTags<
          LinearSolver::Schwarz::Tags::Overlaps<Tags, Dim, OptionsGroup>...,
          LinearSolver::Schwarz::Tags::Overlaps<
              domain::Tags::Interface<
                  domain::Tags::BoundaryDirectionsInterior<Dim>, Tags>,
              Dim, OptionsGroup>...>>(
      std::move(box),
      typename LinearSolver::Schwarz::Tags::Overlaps<Tags, Dim,
                                                     OptionsGroup>::type{}...,
      typename LinearSolver::Schwarz::Tags::Overlaps<
          domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                                  Tags>,
          Dim, OptionsGroup>::type{}...);
}
}

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
template <size_t Dim, typename OptionsGroup, typename BoundaryConditionsTag,
          typename BackgroundFields, typename CommunicatedFields,
          typename PrimalFields, typename InvMetricTag>
struct InitializeSubdomain;

template <size_t Dim, typename OptionsGroup, typename BoundaryConditionsTag,
          typename... BackgroundFields, typename... CommunicatedFields,
          typename PrimalFields, typename InvMetricTag>
struct InitializeSubdomain<
    Dim, OptionsGroup, BoundaryConditionsTag, tmpl::list<BackgroundFields...>,
    tmpl::list<CommunicatedFields...>, PrimalFields, InvMetricTag> {
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
                    const Parallel::GlobalCache<Metavariables>& cache,
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
    const auto& background =
        Parallel::get<typename Metavariables::background_tag>(cache);
    const auto& boundary_conditions_provider =
        db::get<BoundaryConditionsTag>(box);

    overlaps<Mesh<Dim>> overlap_meshes{};
    overlaps<size_t> overlap_extents{};
    overlaps<Element<Dim>> overlap_elements{};
    overlaps<ElementMap<Dim, Frame::Inertial>> overlap_element_maps{};
    overlaps<DirectionMap<Dim, tnsr::i<DataVector, Dim>>>
        overlap_face_normals{};
    overlaps<DirectionMap<Dim, Scalar<DataVector>>>
        overlap_face_normal_magnitudes{};
    overlaps<DirectionMap<Dim, Scalar<DataVector>>> overlap_surface_jacobians{};
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
    overlaps<tnsr::I<DataVector, Dim, Frame::Inertial>>
        overlap_inertial_coords{};
    overlaps<std::unordered_map<
        Direction<Dim>, tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
                            elliptic::Tags::BoundaryCondition, PrimalFields>>>>
        overlap_boundary_condition_types{};
    tuples::TaggedTuple<overlaps_tag<BackgroundFields>...>
        overlap_background_fields{};
    tuples::TaggedTuple<overlaps_tag<CommunicatedFields>...>
        overlap_communicated_fields{};
    overlaps<std::unordered_map<Direction<Dim>,
                                tnsr::I<DataVector, Dim, Frame::Inertial>>>
        overlap_boundary_inertial_coords{};
    tuples::TaggedTuple<overlaps_tag<domain::Tags::Interface<
        domain::Tags::BoundaryDirectionsInterior<Dim>, BackgroundFields>>...>
        overlap_boundary_background_fields{};

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
            LinearSolver::Schwarz::overlap_extent(
                neighbor_mesh.extents(dimension_in_neighbor), max_overlap));
        const size_t overlap_num_points =
            LinearSolver::Schwarz::overlap_num_points(
                neighbor_mesh.extents(), overlap_extents.at(overlap_id),
                dimension_in_neighbor);
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
        // Faces
        DirectionMap<Dim, tnsr::i<DataVector, Dim>> neighbor_face_normals{};
        DirectionMap<Dim, Scalar<DataVector>> neighbor_face_normal_magnitudes{};
        DirectionMap<Dim, Scalar<DataVector>> neighbor_surface_jacobians{};
        const auto setup_face = [&](const Direction<Dim>& local_direction) {
          const auto neighbor_face_mesh =
              neighbor_mesh.slice_away(local_direction.dimension());
          auto neighbor_face_normal = unnormalized_face_normal(
              neighbor_face_mesh, neighbor_element_map, local_direction);
          Scalar<DataVector> neighbor_normal_magnitude{
              neighbor_face_mesh.number_of_grid_points()};
          if constexpr (std::is_same_v<InvMetricTag, void>) {
            magnitude(make_not_null(&neighbor_normal_magnitude),
                      neighbor_face_normal);
          } else {
            const auto face_inertial_coords =
                neighbor_element_map(interface_logical_coordinates(
                    neighbor_face_mesh, local_direction));
            magnitude(make_not_null(&neighbor_normal_magnitude),
                      neighbor_face_normal,
                      get<InvMetricTag>(background.variables(
                          face_inertial_coords, tmpl::list<InvMetricTag>{})));
          }
          for (size_t d = 0; d < Dim; d++) {
            neighbor_face_normal.get(d) /= get(neighbor_normal_magnitude);
          }
          auto neighbor_surface_jacobian = domain::surface_jacobian(
              neighbor_element_map, neighbor_face_mesh, local_direction,
              neighbor_normal_magnitude);
          neighbor_face_normals[local_direction] =
              std::move(neighbor_face_normal);
          neighbor_face_normal_magnitudes[local_direction] =
              std::move(neighbor_normal_magnitude);
          neighbor_surface_jacobians[local_direction] =
              std::move(neighbor_surface_jacobian);
        };
        // Mortars
        ::dg::MortarMap<Dim, Mesh<Dim - 1>> neighbor_mortar_meshes{};
        ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>> neighbor_mortar_sizes{};
        for (const auto& neighbor_direction_and_neighbors :
             neighbor.neighbors()) {
          const auto& neighbor_direction =
              neighbor_direction_and_neighbors.first;
          setup_face(neighbor_direction);
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
        std::unordered_map<
            Direction<Dim>,
            tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
                elliptic::Tags::BoundaryCondition, PrimalFields>>>
            neighbor_boundary_condition_types{};
        overlap_boundary_inertial_coords.emplace(
            overlap_id,
            std::unordered_map<Direction<Dim>,
                               tnsr::I<DataVector, Dim, Frame::Inertial>>{});
        for (const auto& neighbor_direction : neighbor.external_boundaries()) {
          setup_face(neighbor_direction);
          const auto neighbor_mortar_id = std::make_pair(
              neighbor_direction, ElementId<Dim>::external_boundary_id());
          neighbor_mortar_meshes.emplace(
              neighbor_mortar_id,
              neighbor_mesh.slice_away(neighbor_direction.dimension()));
          neighbor_mortar_sizes.emplace(
              neighbor_mortar_id,
              make_array<Dim - 1>(Spectral::MortarSize::Full));
          neighbor_boundary_condition_types[neighbor_direction];
          tmpl::for_each<PrimalFields>([&](auto tag_v) noexcept {
            using tag = tmpl::type_from<decltype(tag_v)>;
            get<elliptic::Tags::BoundaryCondition<tag>>(
                neighbor_boundary_condition_types[direction]) =
                boundary_conditions_provider.boundary_condition_type(
                    overlap_element_maps.at(overlap_id)(
                        interface_logical_coordinates(
                            neighbor_mortar_meshes.at(neighbor_mortar_id),
                            neighbor_direction)),
                    neighbor_direction, tag{});
          });
          overlap_boundary_inertial_coords.at(overlap_id)
              .emplace(neighbor_direction,
                       overlap_element_maps.at(overlap_id)(
                           interface_logical_coordinates(
                               neighbor_mortar_meshes.at(neighbor_mortar_id),
                               neighbor_direction)));
        }
        overlap_face_normals.emplace(overlap_id,
                                     std::move(neighbor_face_normals));
        overlap_face_normal_magnitudes.emplace(
            overlap_id, std::move(neighbor_face_normal_magnitudes));
        overlap_surface_jacobians.emplace(
            overlap_id, std::move(neighbor_surface_jacobians));
        overlap_mortar_meshes.emplace(overlap_id,
                                      std::move(neighbor_mortar_meshes));
        overlap_mortar_sizes.emplace(overlap_id,
                                     std::move(neighbor_mortar_sizes));
        overlap_boundary_condition_types.emplace(
            overlap_id, std::move(neighbor_boundary_condition_types));
        auto neighbor_inertial_coords = overlap_element_maps.at(overlap_id)(
            logical_coordinates(neighbor_mesh));
        overlap_inertial_coords.emplace(overlap_id,
                                        std::move(neighbor_inertial_coords));

        // Neighbor's neighbors
        ::dg::MortarMap<Dim, Mesh<Dim>> neighbors_neighbor_meshes{};
        ::dg::MortarMap<Dim, Scalar<DataVector>>
            neighbors_neighbor_face_normal_magnitudes{};
        ::dg::MortarMap<Dim, Mesh<Dim - 1>> neighbors_neighbor_mortar_meshes{};
        ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>
            neighbors_neighbor_mortar_sizes{};
        for (const auto& neighbor_direction_and_neighbors :
             neighbor.neighbors()) {
          const auto& neighbor_direction =
              neighbor_direction_and_neighbors.first;
          const auto& neighbors_neighbor_orientation =
              neighbor_direction_and_neighbors.second.orientation();
          const auto direction_from_neighbors_neighbor =
              neighbors_neighbor_orientation(neighbor_direction.opposite());
          const auto reoriented_neighbor_face_mesh =
              neighbors_neighbor_orientation(neighbor_mesh)
                  .slice_away(direction_from_neighbors_neighbor.dimension());
          for (const auto& neighbors_neighbor_id :
               neighbor_direction_and_neighbors.second) {
            const ::dg::MortarId<Dim> neighbors_neighbor_mortar_id{
                neighbor_direction, neighbors_neighbor_id};
            neighbors_neighbor_meshes.emplace(
                neighbors_neighbor_mortar_id,
                domain::Initialization::create_initial_mesh(
                    initial_extents, neighbors_neighbor_id));
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
            if constexpr (std::is_same_v<InvMetricTag, void>) {
              magnitude(
                  make_not_null(&neighbors_neighbor_face_normal_magnitude),
                  neighbors_neighbor_face_normal);
            } else {
              const auto face_inertial_coords =
                  neighbors_neighbor_element_map(interface_logical_coordinates(
                      neighbors_neighbor_face_mesh,
                      direction_from_neighbors_neighbor));
              magnitude(
                  make_not_null(&neighbors_neighbor_face_normal_magnitude),
                  neighbors_neighbor_face_normal,
                  get<InvMetricTag>(background.variables(
                      face_inertial_coords, tmpl::list<InvMetricTag>{})));
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

        if constexpr (sizeof...(BackgroundFields) > 0) {
          const auto background_fields =
              ::elliptic::Actions::detail::all_background_fields<
                  tmpl::list<BackgroundFields...>>(
                  background, overlap_inertial_coords.at(overlap_id),
                  neighbor_mesh,
                  neighbor_element_map.inv_jacobian(
                      logical_coordinates(neighbor_mesh)));
          tmpl::for_each<tmpl::list<BackgroundFields...>>(
              [&overlap_id, &background_fields, &overlap_background_fields,
               &overlap_boundary_background_fields, &neighbor,
               &neighbor_mesh](const auto tag_v) noexcept {
                using tag = tmpl::type_from<decltype(tag_v)>;
                get<overlaps_tag<tag>>(overlap_background_fields)
                    .emplace(overlap_id, get<tag>(background_fields));
                get<overlaps_tag<domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsInterior<Dim>, tag>>>(
                    overlap_boundary_background_fields)
                    .emplace(overlap_id,
                             std::unordered_map<Direction<Dim>,
                                                typename tag::type>{});
                for (const auto& direction_from_neighbor :
                     neighbor.external_boundaries()) {
                  get<overlaps_tag<domain::Tags::Interface<
                      domain::Tags::BoundaryDirectionsInterior<Dim>, tag>>>(
                      overlap_boundary_background_fields)
                      .at(overlap_id)
                      .emplace(
                          direction_from_neighbor,
                          data_on_slice(
                              get<overlaps_tag<tag>>(overlap_background_fields)
                                  .at(overlap_id),
                              neighbor_mesh.extents(),
                              direction_from_neighbor.dimension(),
                              index_to_slice_at(neighbor_mesh.extents(),
                                                direction_from_neighbor)));
                }
              });
        }
        if constexpr (sizeof...(CommunicatedFields) > 0) {
          tmpl::for_each<tmpl::list<CommunicatedFields...>>(
              [&overlap_id, overlap_num_points,
               &overlap_communicated_fields](const auto tag_v) noexcept {
                using tag = tmpl::type_from<decltype(tag_v)>;
                if constexpr (tt::is_a_v<Tensor, typename tag::type>) {
                  get<overlaps_tag<tag>>(overlap_communicated_fields)
                      .emplace(
                          overlap_id,
                          make_with_value<typename tag::type>(
                              overlap_num_points,
                              std::numeric_limits<double>::signaling_NaN()));
                } else {
                  get<overlaps_tag<tag>>(overlap_communicated_fields)
                      .emplace(overlap_id, typename tag::type{});
                }
              });
        }
      }  // neighbors in direction
    }    // directions

    auto new_box = ::Initialization::merge_into_databox<
        InitializeSubdomain,
        db::AddSimpleTags<overlaps_tag<domain::Tags::Mesh<Dim>>,
                          overlaps_tag<elliptic::dg::Tags::OverlapExtent>,
                          overlaps_tag<domain::Tags::Element<Dim>>,
                          overlaps_tag<domain::Tags::ElementMap<Dim>>>>(
        std::move(box), std::move(overlap_meshes), std::move(overlap_extents),
        std::move(overlap_elements), std::move(overlap_element_maps));

    auto xnew_box = ::Initialization::merge_into_databox<
        InitializeSubdomain,
        db::AddSimpleTags<overlaps_tag<domain::Tags::Faces<
            Dim,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>>>(
        std::move(new_box), std::move(overlap_face_normals));
    auto bnew_box = ::Initialization::merge_into_databox<
        InitializeSubdomain,
        db::AddSimpleTags<overlaps_tag<domain::Tags::Faces<
            Dim,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>>>(
        std::move(xnew_box), std::move(overlap_face_normal_magnitudes));

    auto cnew_box = ::Initialization::merge_into_databox<
        InitializeSubdomain,
        db::AddSimpleTags<
            overlaps_tag<
                domain::Tags::Faces<Dim, domain::Tags::SurfaceJacobian<
                                             Frame::Logical, Frame::Inertial>>>,
            overlaps_tag<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
            overlaps_tag<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>,
            overlaps_tag<::Tags::NeighborMortars<domain::Tags::Mesh<Dim>, Dim>>,
            overlaps_tag<::Tags::NeighborMortars<
                ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>,
                Dim>>,
            overlaps_tag<
                ::Tags::NeighborMortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
            overlaps_tag<
                ::Tags::NeighborMortars<::Tags::MortarSize<Dim - 1>, Dim>>>>(
        std::move(bnew_box), std::move(overlap_surface_jacobians),
        std::move(overlap_mortar_meshes), std::move(overlap_mortar_sizes),
        std::move(overlap_neighbor_meshes),
        std::move(overlap_neighbor_face_normal_magnitudes),
        std::move(overlap_neighbor_mortar_meshes),
        std::move(overlap_neighbor_mortar_sizes));

    auto anew_box = ::Initialization::merge_into_databox<
        InitializeSubdomain,
        db::AddSimpleTags<
            overlaps_tag<domain::Tags::Coordinates<Dim, Frame::Inertial>>,
            overlaps_tag<domain::Tags::Interface<
                domain::Tags::BoundaryDirectionsExterior<Dim>,
                elliptic::Tags::BoundaryConditions<PrimalFields>>>,
            overlaps_tag<domain::Tags::Interface<
                domain::Tags::BoundaryDirectionsInterior<Dim>,
                domain::Tags::Coordinates<Dim, Frame::Inertial>>>>>(
        std::move(cnew_box), std::move(overlap_inertial_coords),
        std::move(overlap_boundary_condition_types),
        std::move(overlap_boundary_inertial_coords));

    using split_bg = tmpl::split_at<tmpl::list<BackgroundFields...>,
                                    tmpl::integral_constant<unsigned int, 6>>;
    using bg1 = tmpl::front<split_bg>;
    using bg2 = tmpl::back<split_bg>;
    using split_bg2 =
        tmpl::split_at<bg2, tmpl::integral_constant<unsigned int, 6>>;
    using bg3 = tmpl::front<split_bg2>;
    using bg4 = tmpl::back<split_bg2>;
    auto nnew_box1 =
        detail::add_tags<Dim, OptionsGroup>(std::move(anew_box), bg1{});
    auto nnew_box2 =
        detail::add_tags<Dim, OptionsGroup>(std::move(nnew_box1), bg3{});
    auto nnew_box =
        detail::add_tags<Dim, OptionsGroup>(std::move(nnew_box2), bg4{});

    tmpl::for_each<tmpl::list<BackgroundFields...>>([&](const auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      db::mutate<overlaps_tag<tag>,
                 overlaps_tag<domain::Tags::Interface<
                     domain::Tags::BoundaryDirectionsInterior<Dim>, tag>>>(
          make_not_null(&nnew_box), [&](const auto bg, const auto bg_face) {
            *bg = std::move(get<overlaps_tag<tag>>(overlap_background_fields));
            *bg_face = std::move(
                get<overlaps_tag<domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsInterior<Dim>, tag>>>(
                    overlap_boundary_background_fields));
          });
    });

    auto nnnnew_box = ::Initialization::merge_into_databox<
        InitializeSubdomain,
        db::AddSimpleTags<overlaps_tag<CommunicatedFields>...>>(
        std::move(nnew_box), std::move(get<overlaps_tag<CommunicatedFields>>(
                                overlap_communicated_fields))...);

    return std::make_tuple(std::move(nnnnew_box));
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
