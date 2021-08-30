// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Options.hpp"

namespace elliptic::amr {

namespace OptionTags {

struct AmrGroup {
  static std::string name() noexcept { return "Amr"; }
  static constexpr Options::String help{"Adaptive mesh-refinement"};
};

struct TargetErrorReduction {
  static constexpr Options::String help =
      "Each AMR level should lead to a reduction in error by this factor.";
  using type = double;
  using group = AmrGroup;
};

struct IncreaseNumPointsUniformly {
  static constexpr Options::String help =
      "Add this number of points to all elements and dimensions.";
  using type = size_t;
  using group = AmrGroup;
};

}  // namespace OptionTags

namespace Tags {

using Level = Convergence::Tags::IterationId<OptionTags::AmrGroup>;

using HasConverged = Convergence::Tags::HasConverged<OptionTags::AmrGroup>;

template <size_t Dim>
struct ParentMesh : db::SimpleTag {
  using type = std::optional<Mesh<Dim>>;
};

struct TargetErrorReduction : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::TargetErrorReduction>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double value) noexcept {
    return value;
  }
};

struct IncreaseNumPointsUniformly : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::IncreaseNumPointsUniformly>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t value) noexcept {
    return value;
  }
};

}  // namespace Tags
}  // namespace elliptic::amr
