// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/PupStlCpp17.hpp"

namespace LinearSolver::multigrid {

namespace OptionTags {

template <typename OptionsGroup>
struct MaxLevels {
  using type = Options::Auto<size_t>;
  static constexpr Options::String help =
      "Maximum number of levels in the multigrid hierarchy. Includes the "
      "finest grid, i.e. set to '1' to disable multigrids. Set to 'Auto' to "
      "coarsen all the way up to single-element blocks.";
  using group = OptionsGroup;
};

template <typename OptionsGroup>
struct OutputVolumeData {
  using type = bool;
  static constexpr Options::String help = "Record volume data";
  using group = OptionsGroup;
};

}  // namespace OptionTags

namespace Tags {

template <size_t Dim>
struct ChildrenRefinementLevels : db::SimpleTag {
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
struct MaxLevels : db::SimpleTag {
  using type = std::optional<size_t>;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::MaxLevels<OptionsGroup>>;
  static type create_from_options(const type value) noexcept { return value; };
};

template <typename OptionsGroup>
struct OutputVolumeData : db::SimpleTag {
  using type = bool;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::OutputVolumeData<OptionsGroup>>;
  static type create_from_options(const type value) noexcept { return value; };
};

/// The multigrid level. The finest grid is always level 0 and the coarsest grid
/// has the highest level.
struct MultigridLevel : db::SimpleTag {
  using type = size_t;
};

/// Indicates the root of the multigrid hierarchy, i.e. level 0.
struct IsFinestGrid : db::SimpleTag {
  using type = bool;
};

template <size_t Dim>
struct ParentId : db::SimpleTag {
  using type = std::optional<ElementId<Dim>>;
};

template <size_t Dim>
struct ChildIds : db::SimpleTag {
  using type = std::unordered_set<ElementId<Dim>>;
};

template <size_t Dim>
struct ParentMesh : db::SimpleTag {
  using type = std::optional<Mesh<Dim>>;
};

// The following tags are related to volume data output

template <typename OptionsGroup>
struct ObservationId : db::SimpleTag {
  using type = size_t;
};
template <typename Tag>
struct PreSmoothingInitial : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingSource : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingResult : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingResidual : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingInitial : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingSource : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingResult : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingResidual : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename OptionsGroup, typename FieldsTag>
struct VolumeDataForOutput : db::SimpleTag {
  using fields_tags = typename FieldsTag::type::tags_list;
  using type = Variables<
      tmpl::append<db::wrap_tags_in<PreSmoothingInitial, fields_tags>,
                   db::wrap_tags_in<PreSmoothingSource, fields_tags>,
                   db::wrap_tags_in<PreSmoothingResult, fields_tags>,
                   db::wrap_tags_in<PreSmoothingResidual, fields_tags>,
                   db::wrap_tags_in<PostSmoothingInitial, fields_tags>,
                   db::wrap_tags_in<PostSmoothingSource, fields_tags>,
                   db::wrap_tags_in<PostSmoothingResult, fields_tags>,
                   db::wrap_tags_in<PostSmoothingResidual, fields_tags>>>;
};

}  // namespace Tags
}  // namespace LinearSolver::multigrid
