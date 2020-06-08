// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"

namespace LinearSolver::multigrid {

namespace OptionTags {

template <typename OptionsGroup>
struct CoarsestGridPoints : db::SimpleTag {
  using type = size_t;
  static constexpr OptionString help =
      "Minimum number of grid points in each dimension that defines the "
      "coarsest grid.";
  using group = OptionsGroup;
};

}  // namespace OptionTags

namespace Tags {

template <size_t Dim>
struct BaseRefinementLevels : db::SimpleTag {
  using base = domain::Tags::InitialRefinementLevels<Dim>;
  using type = tmpl::type_from<base>;
  static constexpr bool pass_metavariables = base::pass_metavariables;
  using option_tags = typename base::option_tags;
  static constexpr auto create_from_options = base::create_from_options;
};

template <size_t Dim>
struct ParentRefinementLevels : db::SimpleTag {
  using base = domain::Tags::InitialRefinementLevels<Dim>;
  using type = tmpl::type_from<base>;
  static constexpr bool pass_metavariables = base::pass_metavariables;
  using option_tags = typename base::option_tags;
  static constexpr auto create_from_options = base::create_from_options;
};

template <typename OptionsGroup>
struct CoarsestGridPoints : db::SimpleTag {
  using type = size_t;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::CoarsestGridPoints<OptionsGroup>>;
  static type create_from_options(const type value) noexcept { return value; };
};

template <size_t Dim>
struct ParentExtents : db::SimpleTag {
  using base = domain::Tags::InitialExtents<Dim>;
  using type = tmpl::type_from<base>;
  static constexpr bool pass_metavariables = base::pass_metavariables;
  using option_tags = typename base::option_tags;
  static constexpr auto create_from_options = base::create_from_options;
};

template <typename SectionIndexTag>
struct ArraySectionBase : db::BaseTag {};

template <typename SectionIndexTag, typename ArrayComponent>
struct ArraySection : db::SimpleTag, ArraySectionBase<SectionIndexTag> {
  using type =
      CProxySection_AlgorithmArray<ArrayComponent,
                                   typename ArrayComponent::array_index>;
  constexpr static bool pass_metavariables = false;
  using option_tags = tmpl::list<>;
  static type create_from_options() noexcept { return {}; };
};

struct MultigridLevel : db::SimpleTag {
  using type = size_t;
  constexpr static bool pass_metavariables = false;
  using option_tags = tmpl::list<>;
  static type create_from_options() noexcept { return 0; };
};

struct IsFinestLevel : db::SimpleTag {
  using type = bool;
};

template <size_t Dim>
struct ParentElementId : db::SimpleTag {
  using type = std::optional<ElementId<Dim>>;
};

template <size_t Dim>
struct ChildElementIds : db::SimpleTag {
  using type = std::vector<ElementId<Dim>>;
};

template <size_t Dim>
struct ParentMesh : db::SimpleTag {
  using type = Mesh<Dim>;
};

template <size_t Dim, typename OptionsGroup>
struct RestrictionOperator : db::SimpleTag {
  using type = std::array<Matrix, Dim>;
};

template <size_t Dim, typename OptionsGroup>
struct ProlongationOperator : db::SimpleTag {
  using type = std::array<Matrix, Dim>;
};

template <typename Tag>
struct PreSmoothingInitial : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingSource : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingResult : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingResidual : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingInitial : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingSource : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingResult : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingResidual : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};

}  // namespace Tags
}  // namespace LinearSolver::multigrid
