// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "ParallelAlgorithms/LinearSolver/InnerProduct.hpp"

#include "Domain/Mesh.hpp"

namespace LinearSolver {
namespace schwarz_detail {

template <size_t Dim, typename FieldTags>
struct OverlapData {
  static constexpr size_t volume_dim = Dim;
  using field_tags = FieldTags;

  Variables<FieldTags> field_data{};

  Mesh<volume_dim> volume_mesh{};
  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
      inv_jacobian{};
  Scalar<DataVector> magnitude_of_face_normal{};
  Index<volume_dim> overlap_extents{};
  Mesh<volume_dim - 1> mortar_mesh{};
  std::array<Spectral::MortarSize, volume_dim - 1> mortar_size{};

  // OverlapData() = default;
  // OverlapData(const OverlapData&) noexcept = default;
  // OverlapData& operator=(const OverlapData&) noexcept = default;
  // OverlapData(OverlapData&&) noexcept = default;
  // OverlapData& operator=(OverlapData&&) noexcept = default;
  // ~OverlapData() noexcept = default;

  // explicit OverlapData(const size_t num_points) noexcept
  //     : field_data{num_points} {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept {
    p | field_data;
    p | volume_mesh;
    p | inv_jacobian;
    p | magnitude_of_face_normal;
    p | overlap_extents;
    p | mortar_mesh;
    p | mortar_size;
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
          lhs.magnitude_of_face_normal,
          lhs.overlap_extents,
          lhs.mortar_mesh,
          lhs.mortar_size};
}

template <size_t Dim, typename FieldTags>
OverlapData<Dim, FieldTags> operator*(
    const double scalar, const OverlapData<Dim, FieldTags>& data) noexcept {
  return {scalar * data.field_data, data.volume_mesh,
          data.inv_jacobian,        data.magnitude_of_face_normal,
          data.overlap_extents,     data.mortar_mesh,
          data.mortar_size};
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
