// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/range/join.hpp>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Oversample.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
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

/// Actions related to the DG subdomain operator
namespace elliptic::dg::subdomain_operator::Actions {

namespace detail {
// Initialize the geometry of a neighbor into which an overlap extends
template <size_t Dim>
void initialize_overlap_geometry(
    const gsl::not_null<Mesh<Dim>*> vars_mesh,
    const gsl::not_null<Mesh<Dim>*> mesh,
    const gsl::not_null<size_t*> extruding_extent,
    const gsl::not_null<Element<Dim>*> element,
    const gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> inertial_coords,
    const gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        inv_jacobian,
    const gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>*>
        face_inertial_coords_internal,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>*>
        face_inertial_coords_external,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>*>
        face_normals_internal,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>*>
        face_normals_external,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_normal_magnitudes_internal*/,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_normal_magnitudes_external*/,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_jacobians*/,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_meshes,
    const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
        mortar_sizes,
    const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
    /*mortar_jacobians*/,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim>>*> neighbor_meshes,
    const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
    /*neighbor_face_normal_magnitudes*/,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*>
        neighbor_mortar_meshes,
    const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
        neighbor_mortar_sizes,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::vector<std::array<size_t, Dim>>& initial_refinement,
    const Spectral::Quadrature quadrature, const size_t oversample_points,
    const Domain<Dim>& domain, const size_t max_overlap,
    const ElementId<Dim>& element_id,
    const Direction<Dim>& overlap_direction) noexcept {
  // Mesh
  *vars_mesh = domain::Initialization::create_initial_mesh(
      initial_extents, element_id, quadrature);
  *mesh = elliptic::dg::oversample(*vars_mesh, oversample_points);
  // Extruding extent
  *extruding_extent = LinearSolver::Schwarz::overlap_extent(
      vars_mesh->extents(overlap_direction.dimension()), max_overlap);
  // Element
  const auto& block = domain.blocks()[element_id.block_id()];
  *element = domain::Initialization::create_initial_element(element_id, block,
                                                            initial_refinement);
  // Element map
  *element_map = ElementMap<Dim, Frame::Inertial>{
      element_id, block.stationary_map().get_clone()};
  // Coordinates
  const auto logical_coords = logical_coordinates(*mesh);
  *inertial_coords = element_map->operator()(logical_coords);
  // Jacobian
  *inv_jacobian = element_map->inv_jacobian(logical_coords);
  *det_inv_jacobian = determinant(*inv_jacobian);
  // Faces and mortars
  for (const auto& [direction, neighbors] : element->neighbors()) {
    const auto face_mesh = mesh->slice_away(direction.dimension());
    (*face_inertial_coords_internal)[direction] = element_map->operator()(
        interface_logical_coordinates(face_mesh, direction));
    (*face_normals_internal)[direction] =
        unnormalized_face_normal(face_mesh, *element_map, direction);
    const auto& orientation = neighbors.orientation();
    const auto direction_from_neighbor = orientation(direction.opposite());
    const auto reoriented_face_mesh =
        orientation(*mesh).slice_away(direction_from_neighbor.dimension());
    for (const auto& neighbor_id : neighbors) {
      const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
      // Geometry on this side of the mortar
      mortar_meshes->emplace(
          mortar_id,
          ::dg::mortar_mesh(
              face_mesh,
              elliptic::dg::oversample(
                  domain::Initialization::create_initial_mesh(
                      initial_extents, neighbor_id, quadrature, orientation),
                  oversample_points)
                  .slice_away(direction.dimension())));
      mortar_sizes->emplace(
          mortar_id, ::dg::mortar_size(element_id, neighbor_id,
                                       direction.dimension(), orientation));
      // Geometry on the other side of the mortar. These are only needed on
      // mortars to neighbors that are not part of the subdomain, so
      // conditionally skipping this setup is a possible optimization. The
      // computational cost and memory usage is probably irrelevant though.
      const auto& neighbor_mesh =
          neighbor_meshes
              ->emplace(mortar_id,
                        elliptic::dg::oversample(
                            domain::Initialization::create_initial_mesh(
                                initial_extents, neighbor_id, quadrature),
                            oversample_points))
              .first->second;
      const auto neighbor_face_mesh =
          neighbor_mesh.slice_away(direction_from_neighbor.dimension());
      neighbor_mortar_meshes->emplace(
          mortar_id,
          ::dg::mortar_mesh(reoriented_face_mesh, neighbor_face_mesh));
      neighbor_mortar_sizes->emplace(
          mortar_id, ::dg::mortar_size(neighbor_id, element_id,
                                       direction_from_neighbor.dimension(),
                                       orientation.inverse_map()));
    }  // neighbors
  }    // internal directions
  for (const auto& direction : element->external_boundaries()) {
    const auto face_mesh = mesh->slice_away(direction.dimension());
    (*face_inertial_coords_external)[direction] = element_map->operator()(
        interface_logical_coordinates(face_mesh, direction));
    (*face_normals_external)[direction] =
        unnormalized_face_normal(face_mesh, *element_map, direction);
    const auto mortar_id =
        std::make_pair(direction, ElementId<Dim>::external_boundary_id());
    mortar_meshes->emplace(mortar_id, face_mesh);
    mortar_sizes->emplace(mortar_id,
                          make_array<Dim - 1>(Spectral::MortarSize::Full));
  }  // external directions
}
}  // namespace detail

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
  static constexpr bool is_curved =
      not std::is_same_v<typename System::inv_metric_tag, void>;
  static constexpr bool has_background_fields =
      not std::is_same_v<typename System::background_fields, tmpl::list<>>;

  // Shortcuts
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  using face_normal_tag =
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>;
  using face_normal_magnitude_tag =
      ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>;

  using geometry_tags = db::wrap_tags_in<
      overlaps_tag,
      tmpl::list<
          domain::Tags::Mesh<Dim>,
          elliptic::dg::Tags::Oversampled<domain::Tags::Mesh<Dim>>,
          elliptic::dg::subdomain_operator::Tags::ExtrudingExtent,
          domain::Tags::Element<Dim>, domain::Tags::ElementMap<Dim>,
          domain::Tags::Coordinates<Dim, Frame::Inertial>,
          domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
          domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<Dim>,
              domain::Tags::Coordinates<Dim, Frame::Inertial>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<Dim>,
              domain::Tags::Coordinates<Dim, Frame::Inertial>>,
          domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                  face_normal_tag>,
          domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                                  face_normal_tag>,
          domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                  face_normal_magnitude_tag>,
          domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                                  face_normal_magnitude_tag>,
          domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                  domain::Tags::DetSurfaceJacobian<
                                      Frame::Logical, Frame::Inertial>>,
          ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
          ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
          ::Tags::Mortars<
              domain::Tags::DetSurfaceJacobian<Frame::Logical, Frame::Inertial>,
              Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              domain::Tags::Mesh<Dim>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              face_normal_magnitude_tag, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              domain::Tags::Mesh<Dim - 1>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              ::Tags::MortarSize<Dim - 1>, Dim>>>;

  // Only slice those background fields to internal boundaries that are
  // necessary for the DG operator, i.e. the background fields in the
  // System::fluxes_computer::argument_tags
  using fluxes_non_background_args =
      tmpl::list_difference<typename System::fluxes_computer::argument_tags,
                            typename System::background_fields>;
  using background_fields_internal =
      tmpl::list_difference<typename System::fluxes_computer::argument_tags,
                            fluxes_non_background_args>;
  using background_fields_overlap_internal = db::wrap_tags_in<
      overlaps_tag,
      tmpl::transform<
          background_fields_internal,
          make_interface_tag<tmpl::_1,
                             tmpl::pin<domain::Tags::InternalDirections<Dim>>,
                             tmpl::pin<tmpl::list<>>>>>;
  // Slice all background fields to external boundaries for use in boundary
  // conditions
  using background_fields_external = typename System::background_fields;
  using background_fields_overlap_external = db::wrap_tags_in<
      overlaps_tag,
      tmpl::transform<
          background_fields_external,
          make_interface_tag<
              tmpl::_1,
              tmpl::pin<domain::Tags::BoundaryDirectionsInterior<Dim>>,
              tmpl::pin<tmpl::list<>>>>>;

 public:
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>,
                 elliptic::dg::Tags::Quadrature,
                 elliptic::dg::Tags::OversamplingOrder>;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>;
  using simple_tags = tmpl::append<
      geometry_tags,
      tmpl::conditional_t<has_background_fields,
                          tmpl::list<overlaps_tag<::Tags::Variables<
                              typename System::background_fields>>>,
                          tmpl::list<>>,
      background_fields_overlap_internal, background_fields_overlap_external>;
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
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const LinearSolver::Schwarz::OverlapId<Dim> overlap_id{direction,
                                                               neighbor_id};
        // Background-agnostic geometry
        using geometry_argument_tags =
            tmpl::list<domain::Tags::InitialExtents<Dim>,
                       domain::Tags::InitialRefinementLevels<Dim>,
                       elliptic::dg::Tags::Quadrature,
                       elliptic::dg::Tags::OversamplingOrder,
                       domain::Tags::Domain<Dim>,
                       LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>;
        elliptic::util::mutate_apply_at<geometry_tags, geometry_argument_tags,
                                        geometry_argument_tags>(
            detail::initialize_overlap_geometry<Dim>, make_not_null(&box),
            overlap_id, neighbor_id, direction_from_neighbor);
        // Background fields
        if constexpr (has_background_fields) {
          initialize_background_fields(box, overlap_id);
        }
        // Normalize face normals
        normalize_face_normals(box, overlap_id);
      }  // neighbors in direction
    }    // directions

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

 private:
  template <typename DbTagsList>
  static void initialize_background_fields(
      db::DataBox<DbTagsList>& box,
      const LinearSolver::Schwarz::OverlapId<Dim>& overlap_id) noexcept {
    const auto& background = db::get<BackgroundTag>(box);
    DirectionMap<Dim, Variables<typename System::background_fields>>
        face_background_fields{};
    elliptic::util::mutate_apply_at<
        tmpl::list<overlaps_tag<
            ::Tags::Variables<typename System::background_fields>>>,
        db::wrap_tags_in<
            overlaps_tag,
            tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                       domain::Tags::Mesh<Dim>,
                       domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                     Frame::Inertial>,
                       domain::Tags::Element<Dim>>>,
        tmpl::list<>>(
        [&background, &face_background_fields](
            const gsl::not_null<Variables<typename System::background_fields>*>
                background_fields,
            const tnsr::I<DataVector, Dim>& inertial_coords,
            const Mesh<Dim>& mesh,
            const InverseJacobian<DataVector, Dim, Frame::Logical,
                                  Frame::Inertial>& inv_jacobian,
            const Element<Dim>& element) noexcept {
          *background_fields = variables_from_tagged_tuple(
              background.variables(inertial_coords, mesh, inv_jacobian,
                                   typename System::background_fields{}));
          for (const auto& direction :
               boost::join(element.internal_boundaries(),
                           element.external_boundaries())) {
            // Slice the background fields to the face instead of evaluating
            // them on the face coords to avoid re-computing them, and because
            // this is also what the DG operator currently does. The result is
            // the same on Gauss-Lobatto grids, but may need adjusting when
            // adding support for Gauss grids.
            data_on_slice(make_not_null(&face_background_fields[direction]),
                          *background_fields, mesh.extents(),
                          direction.dimension(),
                          index_to_slice_at(mesh.extents(), direction));
          }
        },
        make_not_null(&box), overlap_id);
    // Move face background fields into DataBox
    const auto mutate_assign_face_background_field =
        [&box, &overlap_id, &face_background_fields](
            auto tag_v, auto directions_tag_v,
            const Direction<Dim>& direction) noexcept {
          using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
          using directions_tag = std::decay_t<decltype(directions_tag_v)>;
          db::mutate<
              overlaps_tag<domain::Tags::Interface<directions_tag, tag>>>(
              make_not_null(&box),
              [&face_background_fields, &overlap_id,
               &direction](const auto stored_value) noexcept {
                (*stored_value)[overlap_id][direction] =
                    get<tag>(face_background_fields.at(direction));
              });
        };
    const auto& element =
        db::get<overlaps_tag<domain::Tags::Element<Dim>>>(box).at(overlap_id);
    for (const auto& direction : element.internal_boundaries()) {
      tmpl::for_each<background_fields_internal>(
          [&mutate_assign_face_background_field,
           &direction](auto tag_v) noexcept {
            mutate_assign_face_background_field(
                tag_v, domain::Tags::InternalDirections<Dim>{}, direction);
          });
    }
    for (const auto& direction : element.external_boundaries()) {
      tmpl::for_each<background_fields_external>(
          [&mutate_assign_face_background_field,
           &direction](auto tag_v) noexcept {
            mutate_assign_face_background_field(
                tag_v, domain::Tags::BoundaryDirectionsInterior<Dim>{},
                direction);
          });
    }
  }

  template <typename DbTagsList>
  static void normalize_face_normals(
      db::DataBox<DbTagsList>& box,
      const LinearSolver::Schwarz::OverlapId<Dim>& overlap_id) noexcept {
    // First, multiply the DetInvJacobian with the metric determinant
    if constexpr (is_curved) {
      using inv_metric_tag = overlaps_tag<typename System::inv_metric_tag>;
      elliptic::util::mutate_apply_at<
          tmpl::list<overlaps_tag<
              domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>>,
          tmpl::list<inv_metric_tag>, tmpl::list<>>(
          [](const auto det_inv_jacobian, const auto& inv_metric) noexcept {
            get(*det_inv_jacobian) *= sqrt(get(determinant(inv_metric)));
          },
          make_not_null(&box), overlap_id);
    }
    // Faces of the overlapped element (internal and external)
    const auto& element =
        db::get<overlaps_tag<domain::Tags::Element<Dim>>>(box).at(overlap_id);
    const auto normalize_face_normal = [&box, &overlap_id](
                                           const Direction<Dim>&
                                               local_direction,
                                           auto directions_tag_v) noexcept {
      using directions_tag = std::decay_t<decltype(directions_tag_v)>;
      using face_normal_and_magnitude_tag =
          tmpl::list<overlaps_tag<domain::Tags::Interface<directions_tag,
                                                          face_normal_tag>>,
                     overlaps_tag<domain::Tags::Interface<
                         directions_tag, face_normal_magnitude_tag>>>;
      using inv_metric_tag = overlaps_tag<domain::Tags::Interface<
          directions_tag, typename System::inv_metric_tag>>;
      elliptic::util::mutate_apply_at<
          face_normal_and_magnitude_tag,
          tmpl::conditional_t<is_curved, tmpl::list<inv_metric_tag>,
                              tmpl::list<>>,
          tmpl::list<>>(
          [](const auto face_normal, const auto face_normal_magnitude,
             const auto&... inv_metric) noexcept {
            magnitude(face_normal_magnitude, *face_normal, inv_metric...);
            for (size_t d = 0; d < Dim; ++d) {
              face_normal->get(d) /= get(*face_normal_magnitude);
            }
          },
          make_not_null(&box), std::make_tuple(overlap_id, local_direction));
    };
    for (auto& direction : element.internal_boundaries()) {
      normalize_face_normal(direction, domain::Tags::InternalDirections<Dim>{});
      // Face Jacobians
      elliptic::util::mutate_apply_at<
          tmpl::list<overlaps_tag<
              domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                      domain::Tags::DetSurfaceJacobian<
                                          Frame::Logical, Frame::Inertial>>>>,
          tmpl::list<overlaps_tag<domain::Tags::Interface<
              domain::Tags::InternalDirections<Dim>,
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>>,
          tmpl::list<>>(
          [&direction](const auto face_jacobian,
                       const auto& face_normal_magnitude,
                       const auto& element_map, const auto& mesh) noexcept {
            const auto face_mesh = mesh.slice_away(direction.dimension());
            const auto face_logical_coords =
                interface_logical_coordinates(face_mesh, direction);
            const auto det_jacobian_on_face =
                determinant(element_map.jacobian(face_logical_coords));
            *face_jacobian = face_normal_magnitude;
            get(*face_jacobian) *= get(det_jacobian_on_face);
          },
          make_not_null(&box), std::make_tuple(overlap_id, direction),
          db::get<overlaps_tag<domain::Tags::ElementMap<Dim>>>(box).at(
              overlap_id),
          db::get<overlaps_tag<domain::Tags::Mesh<Dim>>>(box).at(overlap_id));
      for (const auto& neighbor_id : element.neighbors().at(direction)) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        // Mortar Jacobians
        elliptic::util::mutate_apply_at<
            tmpl::list<overlaps_tag<
                ::Tags::Mortars<domain::Tags::DetSurfaceJacobian<
                                    Frame::Logical, Frame::Inertial>,
                                Dim>>>,
            tmpl::list<
                overlaps_tag<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
                overlaps_tag<
                    ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>>,
            tmpl::list<>>(
            [&direction](const auto mortar_jacobian, const auto& mortar_mesh,
                         const auto& mortar_sizes,
                         const auto& element_map) noexcept {
              auto mortar_logical_coords =
                  interface_logical_coordinates(mortar_mesh, direction);
              size_t d_m = 0;
              for (size_t d = 0; d < Dim; ++d) {
                if (d == direction.dimension()) {
                  continue;
                }
                if (mortar_sizes.at(d_m) == Spectral::MortarSize::LowerHalf) {
                  mortar_logical_coords.get(d) -= 1.;
                  mortar_logical_coords.get(d) *= 0.5;
                } else if (mortar_sizes.at(d_m) ==
                           Spectral::MortarSize::UpperHalf) {
                  mortar_logical_coords.get(d) += 1.;
                  mortar_logical_coords.get(d) *= 0.5;
                }
                ++d_m;
              }
              const auto det_jacobian_on_mortar =
                  determinant(element_map.jacobian(mortar_logical_coords));
              const auto inv_jacobian_on_mortar =
                  element_map.inv_jacobian(mortar_logical_coords);
              *mortar_jacobian = magnitude(unnormalized_face_normal(
                  mortar_mesh, inv_jacobian_on_mortar, direction));
              get(*mortar_jacobian) *= get(det_jacobian_on_mortar);
              for (const auto& mortar_size : mortar_sizes) {
                if (mortar_size != Spectral::MortarSize::Full) {
                  get(*mortar_jacobian) *= 0.5;
                }
              }
            },
            make_not_null(&box), std::make_tuple(overlap_id, mortar_id),
            db::get<overlaps_tag<domain::Tags::ElementMap<Dim>>>(box).at(
                overlap_id));
      }
    }
    for (auto& direction : element.external_boundaries()) {
      normalize_face_normal(direction,
                            domain::Tags::BoundaryDirectionsInterior<Dim>{});
    }
    // Faces on the other side of the overlapped element's mortars
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& neighbor_meshes = db::get<overlaps_tag<
        elliptic::dg::subdomain_operator::Tags::NeighborMortars<
            domain::Tags::Mesh<Dim>, Dim>>>(box)
                                      .at(overlap_id);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        const auto neighbor_face_mesh =
            neighbor_meshes.at(mortar_id).slice_away(
                direction_from_neighbor.dimension());
        const auto& neighbor_block = domain.blocks()[neighbor_id.block_id()];
        ElementMap<Dim, Frame::Inertial> neighbor_element_map{
            neighbor_id, neighbor_block.stationary_map().get_clone()};
        const auto neighbor_face_normal = unnormalized_face_normal(
            neighbor_face_mesh, neighbor_element_map, direction_from_neighbor);
        using neighbor_face_normal_magnitudes_tag =
            overlaps_tag<elliptic::dg::subdomain_operator::Tags::
                             NeighborMortars<face_normal_magnitude_tag, Dim>>;
        if constexpr (is_curved) {
          const auto& background = db::get<BackgroundTag>(box);
          const auto neighbor_face_inertial_coords =
              neighbor_element_map(interface_logical_coordinates(
                  neighbor_face_mesh, direction_from_neighbor));
          const auto inv_metric_on_face =
              get<typename System::inv_metric_tag>(background.variables(
                  neighbor_face_inertial_coords,
                  tmpl::list<typename System::inv_metric_tag>{}));
          elliptic::util::mutate_apply_at<
              tmpl::list<neighbor_face_normal_magnitudes_tag>, tmpl::list<>,
              tmpl::list<>>(
              [&neighbor_face_normal, &inv_metric_on_face](
                  const auto neighbor_face_normal_magnitude) noexcept {
                magnitude(neighbor_face_normal_magnitude, neighbor_face_normal,
                          inv_metric_on_face);
              },
              make_not_null(&box), std::make_tuple(overlap_id, mortar_id));
        } else {
          elliptic::util::mutate_apply_at<
              tmpl::list<neighbor_face_normal_magnitudes_tag>, tmpl::list<>,
              tmpl::list<>>(
              [&neighbor_face_normal](
                  const auto neighbor_face_normal_magnitude) noexcept {
                magnitude(neighbor_face_normal_magnitude, neighbor_face_normal);
              },
              make_not_null(&box), std::make_tuple(overlap_id, mortar_id));
        }
      }  // neighbors
    }    // internal directions
  }
};

}  // namespace elliptic::dg::subdomain_operator::Actions
