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

namespace LinearSolver {
namespace schwarz_detail {

/*!
 * \brief Data on an element-centered Schwarz subdomain
 */
template <size_t Dim, typename ElementDataType, typename OverlapDataType>
struct SubdomainData {
  static constexpr size_t volume_dim = Dim;
  using element_data_type = ElementDataType;
  using overlap_data_type = OverlapDataType;

  using MortarId = std::pair<Direction<volume_dim>, ElementId<volume_dim>>;
  using BoundaryDataType =
      FixedHashMap<maximum_number_of_neighbors(volume_dim), MortarId,
                   OverlapDataType, boost::hash<MortarId>>;

  SubdomainData() = default;
  SubdomainData(size_t num_points) noexcept : element_data{num_points} {}
  SubdomainData(const ElementDataType& local_element_data,
                const BoundaryDataType& local_boundary_data) noexcept
      : element_data(local_element_data), boundary_data(local_boundary_data) {}

  ElementDataType element_data{};
  BoundaryDataType boundary_data{};

  template <typename RhsElementData, typename RhsOverlapData>
  SubdomainData& operator+=(
      const SubdomainData<Dim, RhsElementData, RhsOverlapData>& rhs) noexcept {
    element_data += rhs.element_data;
    for (auto& id_and_overlap_data : boundary_data) {
      id_and_overlap_data.second +=
          rhs.boundary_data.at(id_and_overlap_data.first);
    }
    return *this;
  }
  template <typename RhsElementData, typename RhsOverlapData>
  SubdomainData& operator-=(
      const SubdomainData<Dim, RhsElementData, RhsOverlapData>& rhs) noexcept {
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

template <size_t Dim, typename LhsElementData, typename RhsElementData,
          typename LhsOverlapData, typename RhsOverlapData>
decltype(auto) operator-(
    const SubdomainData<Dim, LhsElementData, LhsOverlapData>& lhs,
    const SubdomainData<Dim, RhsElementData, RhsOverlapData>& rhs) noexcept {
  SubdomainData<Dim, LhsElementData, LhsOverlapData> result{
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

template <size_t Dim, typename ElementDataType, typename OverlapDataType>
decltype(auto) operator*(
    const double scalar,
    const SubdomainData<Dim, ElementDataType, OverlapDataType>& data) noexcept {
  SubdomainData<Dim, ElementDataType, OverlapDataType> result{
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

template <size_t Dim, typename LhsElementData, typename RhsElementData,
          typename LhsOverlapData, typename RhsOverlapData>
struct InnerProductImpl<
    schwarz_detail::SubdomainData<Dim, LhsElementData, LhsOverlapData>,
    schwarz_detail::SubdomainData<Dim, RhsElementData, RhsOverlapData>> {
  static double apply(
      const schwarz_detail::SubdomainData<Dim, LhsElementData, LhsOverlapData>&
          lhs,
      const schwarz_detail::SubdomainData<Dim, RhsElementData, RhsOverlapData>&
          rhs) noexcept {
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
