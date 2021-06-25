// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Oversample.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class ElementId;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic::dg::Actions {

namespace InitializeDomain_detail {
template <size_t Dim>
void initialize_geometry(
    const gsl::not_null<Mesh<Dim>*> vars_mesh,
    const gsl::not_null<Mesh<Dim>*> mesh,
    const gsl::not_null<Element<Dim>*> element,
    const gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> inertial_coords,
    const gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        inv_jacobian,
    const gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
    const gsl::not_null<std::unordered_set<Direction<Dim>>*>
        internal_directions,
    const gsl::not_null<std::unordered_set<Direction<Dim>>*>
        external_directions,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Direction<Dim>>*>
        face_directions_internal,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>*>
        face_inertial_coords_internal,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>*>
        face_normals_internal,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_normal_magnitudes_internal*/,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_jacobians*/,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Direction<Dim>>*>
        face_directions_external,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>*>
        face_inertial_coords_external,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>*>
        face_normals_external,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_normal_magnitudes_external*/,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_meshes,
    const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
        mortar_sizes,
    const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
    /*mortar_jacobians*/,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::vector<std::array<size_t, Dim>>& initial_refinement,
    const Spectral::Quadrature quadrature, const size_t oversample_points,
    const Domain<Dim>& domain, const ElementId<Dim>& element_id) noexcept {
  // Mesh
  *vars_mesh = domain::Initialization::create_initial_mesh(
      initial_extents, element_id, quadrature);
  *mesh = elliptic::dg::oversample(*vars_mesh, oversample_points);
  // Element
  const auto& block = domain.blocks()[element_id.block_id()];
  if (block.is_time_dependent()) {
    ERROR(
        "The version of the InitializeDomain action being used is for "
        "elliptic systems which do not have any time-dependence but the "
        "domain creator has set up the domain to have time-dependence.");
  }
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
  *internal_directions = element->internal_boundaries();
  *external_directions = element->external_boundaries();
  for (const auto& [direction, neighbors] : element->neighbors()) {
    const auto face_mesh = mesh->slice_away(direction.dimension());
    (*face_directions_internal)[direction] = direction;
    (*face_inertial_coords_internal)[direction] = element_map->operator()(
        interface_logical_coordinates(face_mesh, direction));
    (*face_normals_internal)[direction] =
        unnormalized_face_normal(face_mesh, *element_map, direction);
    const auto& orientation = neighbors.orientation();
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
    }  // neighbors
  }    // internal directions
  for (const auto& direction : element->external_boundaries()) {
    const auto face_mesh = mesh->slice_away(direction.dimension());
    (*face_directions_external)[direction] = direction;
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
}  // namespace InitializeDomain_detail

/*!
 * \ingroup InitializationGroup
 * \brief Initialize items related to the basic structure of the element
 *
 * GlobalCache:
 * - Uses:
 *   - `Tags::Domain<Dim, Frame::Inertial>`
 * DataBox:
 * - Uses:
 *   - `Tags::InitialExtents<Dim>`
 * - Adds:
 *   - `Tags::Mesh<Dim>`
 *   - `Tags::Element<Dim>`
 *   - `Tags::ElementMap<Dim, Frame::Inertial>`
 *   - `Tags::Coordinates<Dim, Frame::Logical>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `Tags::InverseJacobianCompute<
 *   Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::Logical>>`
 *   - `Tags::DetInvJacobianCompute<Dim, Frame::Logical, Frame::Inertial>`
 *   - `Tags::MinimumGridSpacingCompute<Dim, Frame::Inertial>>`
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename System, typename BackgroundTag>
struct InitializeDomain {
 private:
  static constexpr size_t Dim = System::volume_dim;
  static constexpr bool is_curved =
      not std::is_same_v<typename System::inv_metric_tag, void>;
  static constexpr bool has_background_fields =
      not std::is_same_v<typename System::background_fields, tmpl::list<>>;

  template <typename DirectionsTag>
  using face_tags = tmpl::list<
      domain::Tags::Interface<DirectionsTag, domain::Tags::Direction<Dim>>,
      domain::Tags::Interface<DirectionsTag,
                              domain::Tags::Coordinates<Dim, Frame::Inertial>>,
      domain::Tags::Interface<
          DirectionsTag,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          DirectionsTag,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>;
  using geometry_tags = tmpl::flatten<tmpl::list<
      domain::Tags::Mesh<Dim>,
      elliptic::dg::Tags::Oversampled<domain::Tags::Mesh<Dim>>,
      domain::Tags::Element<Dim>, domain::Tags::ElementMap<Dim>,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>,
      domain::Tags::InternalDirections<Dim>,
      domain::Tags::BoundaryDirectionsInterior<Dim>,
      face_tags<domain::Tags::InternalDirections<Dim>>,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          domain::Tags::DetSurfaceJacobian<Frame::Logical, Frame::Inertial>>,
      face_tags<domain::Tags::BoundaryDirectionsInterior<Dim>>,
      ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
      ::Tags::Mortars<
          domain::Tags::DetSurfaceJacobian<Frame::Logical, Frame::Inertial>,
          Dim>>>;
  using background_tags =
      tmpl::list<::Tags::Variables<typename System::background_fields>,
                 domain::Tags::Interface<
                     domain::Tags::InternalDirections<Dim>,
                     ::Tags::Variables<typename System::background_fields>>,
                 domain::Tags::Interface<
                     domain::Tags::BoundaryDirectionsInterior<Dim>,
                     ::Tags::Variables<typename System::background_fields>>>;

 public:
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>,
                 elliptic::dg::Tags::Quadrature,
                 elliptic::dg::Tags::OversamplingOrder>;
  using simple_tags =
      tmpl::append<geometry_tags,
                   tmpl::conditional_t<has_background_fields, background_tags,
                                       tmpl::list<>>>;
  using compute_tags = tmpl::list<>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& element_id, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    // Initialize geometry
    db::mutate_apply<geometry_tags,
                     tmpl::list<domain::Tags::InitialExtents<Dim>,
                                domain::Tags::InitialRefinementLevels<Dim>,
                                elliptic::dg::Tags::Quadrature,
                                elliptic::dg::Tags::OversamplingOrder,
                                domain::Tags::Domain<Dim>>>(
        InitializeDomain_detail::initialize_geometry<Dim>, make_not_null(&box),
        element_id);
    // Background fields
    if constexpr (has_background_fields) {
      db::mutate_apply<
          background_tags,
          tmpl::list<BackgroundTag,
                     domain::Tags::Coordinates<Dim, Frame::Inertial>,
                     elliptic::dg::Tags::Oversampled<domain::Tags::Mesh<Dim>>,
                     domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                   Frame::Inertial>>>(
          [&element](const auto background_fields,
                     const auto face_background_fields_internal,
                     const auto face_background_fields_external,
                     const auto& background, const auto& inertial_coords,
                     const auto& mesh, const auto& inv_jacobian) noexcept {
            *background_fields = variables_from_tagged_tuple(
                background.variables(inertial_coords, mesh, inv_jacobian,
                                     typename System::background_fields{}));
            ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
                   "Only Gauss-Lobatto quadrature currently implemented for "
                   "slicing background fields.");
            for (const auto& direction : element.internal_boundaries()) {
              data_on_slice(
                  make_not_null(&(*face_background_fields_internal)[direction]),
                  *background_fields, mesh.extents(), direction.dimension(),
                  index_to_slice_at(mesh.extents(), direction));
            }
            for (const auto& direction : element.external_boundaries()) {
              data_on_slice(
                  make_not_null(&(*face_background_fields_external)[direction]),
                  *background_fields, mesh.extents(), direction.dimension(),
                  index_to_slice_at(mesh.extents(), direction));
            }
          },
          make_not_null(&box));
    }
    if constexpr (is_curved) {
      db::mutate_apply<tmpl::list<domain::Tags::DetInvJacobian<
                           Frame::Logical, Frame::Inertial>>,
                       tmpl::list<typename System::inv_metric_tag>>(
          [](const auto det_inv_jacobian, const auto& inv_metric) noexcept {
            get(*det_inv_jacobian) *= sqrt(get(determinant(inv_metric)));
          },
          make_not_null(&box));
    }
    const auto normalize_face_normal = [&box](const Direction<Dim>&
                                                  local_direction,
                                              auto directions_tag_v) noexcept {
      using directions_tag = std::decay_t<decltype(directions_tag_v)>;
      using face_normal_and_magnitude_tag = tmpl::list<
          domain::Tags::Interface<
              directions_tag,
              ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          domain::Tags::Interface<
              directions_tag,
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>;
      using inv_metric_tag =
          domain::Tags::Interface<directions_tag,
                                  typename System::inv_metric_tag>;
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
          make_not_null(&box), std::make_tuple(local_direction));
    };
    for (auto& direction : element.internal_boundaries()) {
      normalize_face_normal(direction, domain::Tags::InternalDirections<Dim>{});
      // Face Jacobians
      elliptic::util::mutate_apply_at<
          tmpl::list<
              domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                      domain::Tags::DetSurfaceJacobian<
                                          Frame::Logical, Frame::Inertial>>>,
          tmpl::list<
              domain::Tags::Interface<
                  domain::Tags::InternalDirections<Dim>,
                  ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
              domain::Tags::ElementMap<Dim>, domain::Tags::Mesh<Dim>>,
          tmpl::list<domain::Tags::ElementMap<Dim>, domain::Tags::Mesh<Dim>>>(
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
          make_not_null(&box), std::make_tuple(direction));
      for (const auto& neighbor_id : element.neighbors().at(direction)) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        // Mortar Jacobians
        elliptic::util::mutate_apply_at<
            tmpl::list<::Tags::Mortars<domain::Tags::DetSurfaceJacobian<
                                           Frame::Logical, Frame::Inertial>,
                                       Dim>>,
            tmpl::list<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                       ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
                       domain::Tags::ElementMap<Dim>>,
            tmpl::list<domain::Tags::ElementMap<Dim>>>(
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
            make_not_null(&box), std::make_tuple(mortar_id));
      }
    }
    for (auto& direction : element.external_boundaries()) {
      normalize_face_normal(direction,
                            domain::Tags::BoundaryDirectionsInterior<Dim>{});
    }
    return std::make_tuple(std::move(box));
  }
};
}  // namespace elliptic::dg::Actions
