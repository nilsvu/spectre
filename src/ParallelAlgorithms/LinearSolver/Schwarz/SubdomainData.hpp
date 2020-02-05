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
    p | mortar_mesh;
    p | mortar_size;
  }
};

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
    return *this;
  }
  template <typename RhsTagsList>
  SubdomainData& operator-=(
      const SubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    element_data -= rhs.element_data;
    return *this;
  }
  SubdomainData& operator/=(const double scalar) noexcept {
    element_data /= scalar;
    return *this;
  }
};

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
decltype(auto) operator-(const SubdomainData<Dim, LhsTagsList>& lhs,
                         const SubdomainData<Dim, RhsTagsList>& rhs) noexcept {
  SubdomainData<Dim, LhsTagsList> result{
      lhs.element_data.number_of_grid_points()};
  result.element_data = lhs.element_data - rhs.element_data;
  return result;
}

template <size_t Dim, typename TagsList>
decltype(auto) operator*(const double scalar,
                         const SubdomainData<Dim, TagsList>& data) noexcept {
  SubdomainData<Dim, TagsList> result{
      data.element_data.number_of_grid_points()};
  result.element_data = scalar * data.element_data;
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
    return inner_product(lhs.element_data, rhs.element_data);
  }
};

}  // namespace InnerProductImpls

}  // namespace LinearSolver
