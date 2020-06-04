// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <pup.h>
#include <string>
#include <tuple>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace LinearSolver::Schwarz {

template <size_t Dim, typename TagsList>
struct ElementCenteredSubdomainData {
  static constexpr size_t volume_dim = Dim;
  using ElementData = Variables<TagsList>;
  using OverlapData = ElementData;

  ElementCenteredSubdomainData() = default;
  ElementCenteredSubdomainData(const ElementCenteredSubdomainData&) noexcept =
      default;
  ElementCenteredSubdomainData& operator=(
      const ElementCenteredSubdomainData&) noexcept = default;
  ElementCenteredSubdomainData(ElementCenteredSubdomainData&&) noexcept =
      default;
  ElementCenteredSubdomainData& operator=(
      ElementCenteredSubdomainData&&) noexcept = default;
  ~ElementCenteredSubdomainData() noexcept = default;

  explicit ElementCenteredSubdomainData(
      const size_t element_num_points) noexcept
      : element_data{element_num_points}, overlap_data{} {}

  ElementCenteredSubdomainData(
      Variables<TagsList> local_element_data,
      OverlapMap<Dim, Variables<TagsList>> local_overlap_data) noexcept
      : element_data{std::move(local_element_data)},
        overlap_data{std::move(local_overlap_data)} {}

  void pup(PUP::er& p) noexcept {
    p | element_data;
    p | overlap_data;
  }

  template <typename RhsTagsList>
  ElementCenteredSubdomainData& operator+=(
      const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    element_data += rhs.element_data;
    for (auto& id_and_overlap_data : overlap_data) {
      id_and_overlap_data.second +=
          rhs.overlap_data.at(id_and_overlap_data.first);
    }
    return *this;
  }

  template <typename RhsTagsList>
  ElementCenteredSubdomainData& operator-=(
      const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    element_data -= rhs.element_data;
    for (auto& id_and_overlap_data : overlap_data) {
      id_and_overlap_data.second -=
          rhs.overlap_data.at(id_and_overlap_data.first);
    }
    return *this;
  }

  ElementCenteredSubdomainData& operator*=(const double scalar) noexcept {
    element_data *= scalar;
    for (auto& id_and_overlap_data : overlap_data) {
      id_and_overlap_data.second *= scalar;
    }
    return *this;
  }

  ElementCenteredSubdomainData& operator/=(const double scalar) noexcept {
    element_data /= scalar;
    for (auto& id_and_overlap_data : overlap_data) {
      id_and_overlap_data.second /= scalar;
    }
    return *this;
  }

  ElementData element_data{};
  OverlapMap<Dim, OverlapData> overlap_data{};
};

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
decltype(auto) operator+(
    ElementCenteredSubdomainData<Dim, LhsTagsList> lhs,
    const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
  lhs += rhs;
  return lhs;
}

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
decltype(auto) operator-(
    ElementCenteredSubdomainData<Dim, LhsTagsList> lhs,
    const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
  lhs -= rhs;
  return lhs;
}

template <size_t Dim, typename TagsList>
decltype(auto) operator*(
    const double scalar,
    ElementCenteredSubdomainData<Dim, TagsList> data) noexcept {
  data *= scalar;
  return data;
}

template <size_t Dim, typename TagsList>
decltype(auto) operator/(
    const double scalar,
    ElementCenteredSubdomainData<Dim, TagsList> data) noexcept {
  data /= scalar;
  return data;
}

template <size_t Dim, typename TagsList>
std::ostream& operator<<(std::ostream& os,
                         const ElementCenteredSubdomainData<Dim, TagsList>&
                             subdomain_data) noexcept {
  os << "Element data:\n"
     << subdomain_data.element_data << "\nOverlap data:\n"
     << subdomain_data.overlap_data;
  return os;
}

}  // namespace LinearSolver::Schwarz

namespace LinearSolver::InnerProductImpls {

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
struct InnerProductImpl<
    Schwarz::ElementCenteredSubdomainData<Dim, LhsTagsList>,
    Schwarz::ElementCenteredSubdomainData<Dim, RhsTagsList>> {
  static double apply(
      const Schwarz::ElementCenteredSubdomainData<Dim, LhsTagsList>& lhs,
      const Schwarz::ElementCenteredSubdomainData<Dim, RhsTagsList>&
          rhs) noexcept {
    double result = inner_product(lhs.element_data, rhs.element_data);
    for (const auto& id_and_lhs_overlap_data : lhs.overlap_data) {
      result +=
          inner_product(id_and_lhs_overlap_data.second,
                        rhs.overlap_data.at(id_and_lhs_overlap_data.first));
    }
    return result;
  }
};

}  // namespace LinearSolver::InnerProductImpls

namespace MakeWithValueImpls {

template <size_t Dim, typename TagsListOut, typename TagsListIn>
struct MakeWithValueImpl<
    LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListOut>,
    LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListIn>> {
  using SubdomainDataIn =
      LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListIn>;
  using SubdomainDataOut =
      LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListOut>;
  /// \brief Returns a
  /// `LinearSolver::Schwarz::ElementCenteredSubdomainData`
  /// the same size as `input`, with each element equal to `value`.
  static SPECTRE_ALWAYS_INLINE SubdomainDataOut
  apply(const SubdomainDataIn& input, const double value) noexcept {
    SubdomainDataOut output{input.element_data.number_of_grid_points()};
    output.element_data =
        make_with_value<typename SubdomainDataOut::ElementData>(
            input.element_data, value);
    for (const auto& overlap_id_and_data : input.overlap_data) {
      output.overlap_data.emplace(
          overlap_id_and_data.first,
          make_with_value<typename SubdomainDataOut::OverlapData>(
              overlap_id_and_data.second, value));
    }
    return output;
  }
};

}  // namespace MakeWithValueImpls
