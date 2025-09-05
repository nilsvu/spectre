// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Options/Auto.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace elliptic::dg::subdomain_operator::OptionTags {

/// Impose these boundary conditions on the subdomain instead of clamping the
/// fields to zero outside the subdomain. Only works if `MaxOverlap: Auto` so
/// overlap regions cover the full neighbor.
template <size_t Dim, typename BoundaryConditionsBase, typename OptionsGroup>
struct SubdomainBoundaryConditions : db::SimpleTag {
 private:
  using BoundaryConditionsOptionType = typename domain::creators::Rectilinear<
      Dim>::template BoundaryConditions<BoundaryConditionsBase>::type;

 public:
  static constexpr Options::String help =
      "Impose these boundary conditions on the subdomain instead of clamping "
      "the fields to zero outside the subdomain. Only works if 'MaxOverlap: "
      "Auto' so overlap regions cover the full neighbor.";
  using type = Options::Auto<std::vector<BoundaryConditionsOptionType>,
                             Options::AutoLabel::None>;
  using group = OptionsGroup;
};

}  // namespace elliptic::dg::subdomain_operator::OptionTags

namespace elliptic::dg::subdomain_operator::Tags {

/// The number of points an element-centered subdomain extends into the
/// neighbor, i.e. the "extruding" overlap extents. This tag is used in
/// conjunction with `LinearSolver::Schwarz::Tags::Overlaps` to describe the
/// extruding extent into each neighbor.
struct ExtrudingExtent : db::SimpleTag {
  using type = size_t;
};

/// Data on the neighbor's side of a mortar. Used to store data for elements
/// that do not overlap with the element-centered subdomain, but play a role
/// in the DG operator nonetheless.
template <typename Tag, size_t VolumeDim>
struct NeighborMortars : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = DirectionalIdMap<VolumeDim, typename Tag::type>;
};

/// Impose these boundary conditions on the subdomain instead of clamping the
/// fields to zero outside the subdomain. Only works if `MaxOverlap: Auto` so
/// overlap regions cover the full neighbor.
template <size_t Dim, typename BoundaryConditionsBase, typename OptionsGroup>
struct SubdomainBoundaryConditions : db::SimpleTag {
 private:
  using BoundaryConditionsOptionType = typename domain::creators::Rectilinear<
      Dim>::template BoundaryConditions<BoundaryConditionsBase>::type;

 public:
  using type = std::optional<std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>>;
  using option_tags =
      tmpl::list<OptionTags::SubdomainBoundaryConditions<
                     Dim, BoundaryConditionsBase, OptionsGroup>,
                 LinearSolver::Schwarz::OptionTags::MaxOverlap<OptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::optional<std::vector<BoundaryConditionsOptionType>>& created,
      const std::optional<size_t>& max_overlap) {
    if (not created.has_value()) {
      return std::nullopt;
    }
    if (max_overlap.has_value()) {
      ERROR(
          "Subdomain boundary conditions only work if overlaps cover the full "
          "neighbor. Set 'MaxOverlap: Auto' or disable subdomain boundary "
          "conditions.");
    }
    typename type::value_type result;
    for (const auto& block_boundary_conditions : created.value()) {
      auto lower_upper_bcs =
          domain::creators::Rectilinear<Dim>::transform_boundary_conditions(
              block_boundary_conditions);
      DirectionMap<
          Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
          direction_map;
      for (size_t dim = 0; dim < Dim; ++dim) {
        direction_map[Direction<Dim>{dim, Side::Lower}] =
            std::move(lower_upper_bcs[dim][0]);
        direction_map[Direction<Dim>{dim, Side::Upper}] =
            std::move(lower_upper_bcs[dim][1]);
      }
      result.emplace_back(std::move(direction_map));
    }
    return result;
  }
};

}  // namespace elliptic::dg::subdomain_operator::Tags
