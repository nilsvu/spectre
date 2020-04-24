// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainHelpers.hpp"

#include "Domain/LogicalCoordinates.hpp"

/*!
 * \brief Data on a region within an element that extends a certain number of
 * grid points from a face into the volume.
 */
template <size_t Dim, typename FieldTags>
struct OverlapData {
  static constexpr size_t volume_dim = Dim;
  using field_tags = FieldTags;

  // Variable quantities
  Variables<FieldTags> field_data{};

  // Geometric quantities
  Mesh<volume_dim> volume_mesh{};
  ElementMap<volume_dim, Frame::Inertial> element_map{};
  /// Direction from which the overlap extends into the overlapped element
  Direction<volume_dim> direction{};
  Index<volume_dim> overlap_extents{};
  ::dg::MortarMap<volume_dim, Mesh<volume_dim - 1>>
      perpendicular_mortar_meshes{};
  ::dg::MortarMap<volume_dim, ::dg::MortarSize<volume_dim - 1>>
      perpendicular_mortar_sizes{};

  size_t overlap_extent() const noexcept {
    return overlap_extents[direction.dimension()];
  }

  double overlap_width() const noexcept {
    return LinearSolver::schwarz_detail::overlap_width(
        volume_mesh.slice_through(direction.dimension()), overlap_extent());
  }

  tnsr::I<DataVector, volume_dim, Frame::Logical> logical_coordinates() const
      noexcept {
    return restrict_to_overlap(::logical_coordinates(volume_mesh),
                               volume_mesh.extents(), overlap_extents,
                               direction);
  }

  // OverlapData() = default;
  // OverlapData(const OverlapData&) noexcept = default;
  // OverlapData& operator=(const OverlapData&) noexcept = default;
  // OverlapData(OverlapData&&) noexcept = default;
  // OverlapData& operator=(OverlapData&&) noexcept = default;
  // ~OverlapData() noexcept = default;

  // explicit OverlapData(const size_t num_points) noexcept
  //     : field_data{num_points} {}

  /*!
   * \brief Extend the field data to the full mesh by filling it with zeros and
   * adding the overlapping slices.
   *
   * Note that `direction` must be consistent with the `volume_mesh` and the
   * `overlap_extents` in the sense that the `direction.dimension()` is the
   * overlapping dimension. This means the data must be oriented correctly.
   */
  Variables<FieldTags> extended_field_data() const noexcept {
    return extended_overlap_data(field_data, volume_mesh.extents(),
                                 overlap_extents, direction);
  }

  void orient(const OrientationMap<Dim>& orientation) noexcept {
    if (orientation.is_aligned()) {
      return;
    }
    field_data = orient_variables(field_data, overlap_extents, orientation);
    // inv_jacobian =
    //     orient_tensor(inv_jacobian, volume_mesh.extents(), orientation);
    // magnitude_of_face_normal = orient_tensor_on_slice(
    //     magnitude_of_face_normal,
    //     volume_mesh.slice_away(direction.dimension()).extents(),
    //     direction.dimension(), orientation);

    // for (auto& mortar_id_and_mesh : perpendicular_mortar_meshes) {
    //   const auto& mortar_id = mortar_id_and_mesh.first;
    //   auto& mortar_mesh = mortar_id_and_mesh.second;
    //   auto& mortar_size = perpendicular_mortar_sizes[mortar_id];
    //   const auto& face_direction = mortar_id.first;
    //   const auto orientation_on_face =
    //       orientation.slice_away(face_direction.dimension());
    //   mortar_mesh = Mesh<Dim - 1>{orientation_on_face(mortar_mesh)};
    //   mortar_size = orientation_on_face.permute_from_neighbor(mortar_size);
    // }

    // Orient these quantities last because previous calls may use them
    volume_mesh = orientation(volume_mesh);
    direction = orientation(direction);
    overlap_extents = Index<volume_dim>{
        orientation.permute_from_neighbor(overlap_extents.indices())};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept {
    p | field_data;
    p | volume_mesh;
    p | element_map;
    p | direction;
    p | overlap_extents;
    p | perpendicular_mortar_meshes;
    p | perpendicular_mortar_sizes;
  }

  template <typename RhsFieldTags>
  OverlapData& operator+=(const OverlapData<Dim, RhsFieldTags>& rhs) noexcept {
    field_data += rhs.field_data;
    return *this;
  }
  template <typename RhsFieldTags>
  OverlapData& operator-=(const OverlapData<Dim, RhsFieldTags>& rhs) noexcept {
    field_data -= rhs.field_data;
    return *this;
  }
  OverlapData& operator/=(const double scalar) noexcept {
    field_data /= scalar;
    return *this;
  }
};

template <size_t Dim, typename LhsFieldTags, typename RhsFieldTags>
OverlapData<Dim, LhsFieldTags> operator-(
    const OverlapData<Dim, LhsFieldTags>& lhs,
    const OverlapData<Dim, RhsFieldTags>& rhs) noexcept {
  return {lhs.field_data - rhs.field_data,
          lhs.volume_mesh,
          lhs.element_map,
          lhs.direction,
          lhs.overlap_extents,
          lhs.perpendicular_mortar_meshes,
          lhs.perpendicular_mortar_sizes};
}

template <size_t Dim, typename FieldTags>
OverlapData<Dim, FieldTags> operator*(
    const double scalar, const OverlapData<Dim, FieldTags>& data) noexcept {
  return {scalar * data.field_data,
          data.volume_mesh,
          data.element_map,
          data.direction,
          data.overlap_extents,
          data.perpendicular_mortar_meshes,
          data.perpendicular_mortar_sizes};
}

namespace InnerProductImpls {

template <size_t Dim, typename LhsFieldTags, typename RhsFieldTags>
struct InnerProductImpl<schwarz_detail::OverlapData<Dim, LhsFieldTags>,
                        schwarz_detail::OverlapData<Dim, RhsFieldTags>> {
  static double apply(
      const schwarz_detail::OverlapData<Dim, LhsFieldTags>& lhs,
      const schwarz_detail::OverlapData<Dim, RhsFieldTags>& rhs) noexcept {
    return inner_product(lhs.field_data, rhs.field_data);
  }
};

}  // namespace InnerProductImpls

template <size_t Dim, typename FieldsTag, typename TagsList,
          typename OptionsGroup>
struct CollectOverlapData {
  using argument_tags =
      tmpl::list<FieldsTag, domain::Tags::Mesh<Dim>,
                 domain::Tags::ElementMap<Dim>, domain::Tags::Direction<Dim>,
                 Tags::Overlap<OptionsGroup>,
                 ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;
  using volume_tags =
      tmpl::list<FieldsTag, domain::Tags::Mesh<Dim>,
                 domain::Tags::ElementMap<Dim>, Tags::Overlap<OptionsGroup>,
                 ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;
  static auto apply(
      const db::const_item_type<FieldsTag>& fields, const Mesh<Dim>& mesh,
      const ElementMap<Dim>& element_map, const Direction<Dim>& direction,
      const size_t overlap,
      const db::const_item_type<
          ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>& mortar_meshes,
      const db::const_item_type<
          ::Tags::Mortars<domain::Tags::MortarSize<Dim - 1>, Dim>>&
          mortar_sizes) noexcept {
    const size_t dimension = direction.dimension();
    const auto overlap_extents = LinearSolver::schwarz_detail::overlap_extents(
        mesh.extents(), overlap, dimension);
    const auto fields_on_overlap =
        data_on_overlap(fields, mesh.extents(), overlap_extents, direction);
    const auto perpendicular_mortar_meshes =
        perpendicular(mortar_meshes, direction);
    const auto perpendicular_mortar_sizes =
        perpendicular(mortar_sizes, direction);
    return OverlapData<Dim, FieldTags>{Variables<TagsList>{fields_on_overlap},
                                       mesh,
                                       element_map,
                                       direction,
                                       overlap_extents,
                                       perpendicular_mortar_meshes,
                                       perpendicular_mortar_sizes};
  }
};

struct Weighting {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Logical>>;
  template <typename SubdomainData>
  void apply(
      const tnsr::I<DataVector, Dim, Frame::Logical>& central_logical_coords,
      const gsl::not_null<SubdomainData*> subdomain_data) noexcept {
    // TODO: The central element will receive overlap contributions from its
    // face neighbors, so should we weight the subdomain solution with each
    // neighbor's _incoming_ overlap width?
    // TODO: Is this the correct way to handle h-refined mortars?
    // TODO: The overlap width we'll receive may be different to the overlap
    // we're sending because of p-refinement. Should we weight with the expected
    // incoming contribution's width or with the one we're sending?
    // TODO: We'll have to keep in mind that the weighting operation should
    // preserve symmetry of the linear operator
    for (auto& mortar_id_and_overlap : subdomain_data->boundary_data) {
      const auto& mortar_id = mortar_id_and_overlap.first;
      auto& overlap_data = mortar_id_and_overlap.second;
      const auto& direction = mortar_id.first;
      const size_t dimension = direction.dimension();
      const auto& central_logical_coord_in_dim =
          central_logical_coords.get(dimension);
      // Use incoming or outgoing overlap width here?
      // const double overlap_width = overlap_width(
      //     mesh.slice_through(dimension),
      //     overlap_extent(
      //         mesh.extents(dimension),
      //         get<LinearSolver::Tags::Overlap<OptionsGroup>>(box)));
      const double overlap_width = overlap_data.overlap_width();
      // Parallel::printf(
      //     "%s  Weighting center with width %f for overlap with %s\n",
      //     element_index, overlap_width_in_center, mortar_id);
      // Parallel::printf("%s  Logical coords for overlap with %s: %s\n",
      //                  element_index, mortar_id, logical_coord);
      const auto weight_in_central_element =
          weight(central_logical_coord_in_dim, overlap_width, direction.side());
      // Parallel::printf("%s  Weights:\n%s\n", element_index, w);
      subdomain_data->element_data *= weight_in_central_element;

      const auto neighbor_logical_coords = overlap_data.logical_coordinates();

      const DataVector extended_logical_coord_in_dim =
          neighbor_logical_coords.get(dimension) + direction.sign() * 2.;
      overlap_data.field_data *= weight(extended_logical_coord_in_dim,
                                        overlap_width, direction.side());
      for (const auto& other_mortar_id_and_overlap :
           subdomain_data->boundary_data) {
        const auto& other_mortar_id = other_mortar_id_and_overlap.first;
        const auto& other_direction = other_mortar_id.first;
        const size_t other_dim = other_direction.dimension();
        if (other_dim == dimension) {
          // Neither other (h-refined) overlaps on this side nor on the opposite
          // side contribute to this overlap's weighting
          continue;
        }
        const auto& other_overlap_data = other_mortar_id_and_overlap.second;
        const double other_overlap_width = other_overlap_data.overlap_width();
        const auto& logical_coord_in_other_dim =
            neighbor_logical_coords.get(other_dim);
        overlap_data.field_data *=
            weight(logical_coord_in_other_dim, other_overlap_width,
                   other_direction.side());
      }
    }
  }
