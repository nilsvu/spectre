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
#include "ParallelAlgorithms/LinearSolver/InnerProduct.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainHelpers.hpp"

namespace LinearSolver {
namespace schwarz_detail {

template <size_t Dim, typename FieldTags>
struct OverlapData {
  static constexpr size_t volume_dim = Dim;
  using field_tags = FieldTags;

  // Variable quantities
  Variables<FieldTags> field_data{};

  // Geometric quantities
  Mesh<volume_dim> volume_mesh{};
  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
      inv_jacobian{};
  Direction<volume_dim> direction{};
  Scalar<DataVector> magnitude_of_face_normal{};
  Index<volume_dim> overlap_extents{};

  size_t overlap() const noexcept {
    return overlap_extents[direction.dimension()];
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
    inv_jacobian =
        orient_tensor(inv_jacobian, volume_mesh.extents(), orientation);
    magnitude_of_face_normal = orient_tensor_on_slice(
        magnitude_of_face_normal,
        volume_mesh.slice_away(direction.dimension()).extents(),
        direction.dimension(), orientation);
    // const auto orientation_on_face =
    //     orientation.slice_away(direction.dimension());
    // mortar_mesh = orientation_on_face(mortar_mesh);
    // mortar_size = orientation_on_face.permute_from_neighbor(mortar_size);
    // Orient these quantities last because previous calls are using them
    volume_mesh = orientation(volume_mesh);
    direction = orientation(direction);
    overlap_extents = Index<volume_dim>{
        orientation.permute_from_neighbor(overlap_extents.indices())};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept {
    p | field_data;
    p | volume_mesh;
    p | inv_jacobian;
    p | direction;
    p | magnitude_of_face_normal;
    p | overlap_extents;
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
          lhs.inv_jacobian,
          lhs.direction,
          lhs.magnitude_of_face_normal,
          lhs.overlap_extents};
}

template <size_t Dim, typename FieldTags>
OverlapData<Dim, FieldTags> operator*(
    const double scalar, const OverlapData<Dim, FieldTags>& data) noexcept {
  return {scalar * data.field_data,
          data.volume_mesh,
          data.inv_jacobian,
          data.direction,
          data.magnitude_of_face_normal,
          data.overlap_extents};
}

}  // namespace schwarz_detail

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

namespace schwarz_detail {

/*!
 * \brief Data on an element-centered Schwarz subdomain
 */
template <size_t Dim, typename TagsList>
struct SubdomainData {
  static constexpr size_t volume_dim = Dim;
  using Vars = Variables<TagsList>;
  using MortarId = std::pair<Direction<volume_dim>, ElementId<volume_dim>>;
  using OverlapDataType = OverlapData<volume_dim, TagsList>;
  using BoundaryDataType =
      FixedHashMap<maximum_number_of_neighbors(volume_dim), MortarId,
                   OverlapDataType, boost::hash<MortarId>>;

  SubdomainData() = default;
  SubdomainData(size_t num_points) noexcept : element_data{num_points} {}
  SubdomainData(const Vars& local_element_data,
                const BoundaryDataType& local_boundary_data) noexcept
      : element_data(local_element_data), boundary_data(local_boundary_data) {}

  Vars element_data{};
  BoundaryDataType boundary_data{};

  // TODO: Add boundary contributions to all operators
  template <typename RhsTagsList>
  SubdomainData& operator+=(
      const SubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    element_data += rhs.element_data;
    for (auto& id_and_overlap_data : boundary_data) {
      id_and_overlap_data.second +=
          rhs.boundary_data.at(id_and_overlap_data.first);
    }
    return *this;
  }
  template <typename RhsTagsList>
  SubdomainData& operator-=(
      const SubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    element_data -= rhs.element_data;
    for (auto& id_and_overlap_data : boundary_data) {
      id_and_overlap_data.second -=
          rhs.boundary_data.at(id_and_overlap_data.first);
    }
    return *this;
  }
  SubdomainData& operator/=(const double scalar) noexcept {
    element_data /= scalar;
    for (auto& id_and_overlap_data : boundary_data) {
      id_and_overlap_data.second /= scalar;
    }
    return *this;
  }
};

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
decltype(auto) operator-(const SubdomainData<Dim, LhsTagsList>& lhs,
                         const SubdomainData<Dim, RhsTagsList>& rhs) noexcept {
  SubdomainData<Dim, LhsTagsList> result{
      lhs.element_data.number_of_grid_points()};
  result.element_data = lhs.element_data - rhs.element_data;
  for (const auto& id_and_lhs_overlap_data : lhs.boundary_data) {
    result.boundary_data.emplace(
        id_and_lhs_overlap_data.first,
        id_and_lhs_overlap_data.second -
            rhs.boundary_data.at(id_and_lhs_overlap_data.first));
  }
  return result;
}

template <size_t Dim, typename TagsList>
decltype(auto) operator*(const double scalar,
                         const SubdomainData<Dim, TagsList>& data) noexcept {
  SubdomainData<Dim, TagsList> result{
      data.element_data.number_of_grid_points()};
  result.element_data = scalar * data.element_data;
  for (const auto& id_and_overlap_data : data.boundary_data) {
    result.boundary_data.emplace(id_and_overlap_data.first,
                                 scalar * id_and_overlap_data.second);
  }
  return result;
}

}  // namespace schwarz_detail

namespace InnerProductImpls {

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
struct InnerProductImpl<schwarz_detail::SubdomainData<Dim, LhsTagsList>,
                        schwarz_detail::SubdomainData<Dim, RhsTagsList>> {
  static double apply(
      const schwarz_detail::SubdomainData<Dim, LhsTagsList>& lhs,
      const schwarz_detail::SubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    double result = inner_product(lhs.element_data, rhs.element_data);
    for (const auto& id_and_lhs_overlap_data : lhs.boundary_data) {
      result +=
          inner_product(id_and_lhs_overlap_data.second,
                        rhs.boundary_data.at(id_and_lhs_overlap_data.first));
    }
    return result;
  }
};

}  // namespace InnerProductImpls

}  // namespace LinearSolver
