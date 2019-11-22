// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"

namespace OptionTags {

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution {
  static constexpr OptionString help = "Options for the analytic solution";
  using type = std::unique_ptr<SolutionType>;
};

/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition {
  static constexpr OptionString help = "Boundary condition to be used";
  using type = BoundaryConditionType;
};
}  // namespace OptionTags

namespace Tags {
/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticSolution.
struct AnalyticSolutionBase : db::BaseTag {};

/// Base tag with which to retrieve the BoundaryConditionType
struct BoundaryConditionBase : db::BaseTag {};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution : AnalyticSolutionBase, db::SimpleTag {
  using type = std::unique_ptr<SolutionType>;
  using option_tags = tmpl::list<::OptionTags::AnalyticSolution<SolutionType>>;

  template <typename Metavariables>
  static type create_from_options(const type& analytic_solution) noexcept {
    return deserialize<type>(serialize<type>(analytic_solution).data());
  }
};

/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition : BoundaryConditionBase, db::SimpleTag {
  static std::string name() noexcept { return "BoundaryCondition"; }
  using type = BoundaryConditionType;
  using option_tags =
      tmpl::list<::OptionTags::BoundaryCondition<BoundaryConditionType>>;

  template <typename Metavariables>
  static BoundaryConditionType create_from_options(
      const BoundaryConditionType& boundary_condition) noexcept {
    return boundary_condition;
  }
};
}  // namespace Tags
