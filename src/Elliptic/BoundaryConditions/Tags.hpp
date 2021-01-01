// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/Options.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic {
namespace OptionTags {

template <typename Tag>
struct BoundaryConditionType {
  static std::string name() noexcept { return db::tag_name<Tag>(); }
  using type = elliptic::BoundaryConditionType;
  static constexpr Options::String help =
      "Type of boundary conditions to impose on this variable";
};
}  // namespace OptionTags

namespace Tags {

template <typename Tag>
struct BoundaryConditionType : db::PrefixTag {
  using type = elliptic::BoundaryConditionType;
};

template <typename Tags>
struct BoundaryConditionTypes : db::SimpleTag {
  using type = tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<elliptic::Tags::BoundaryConditionType, Tags>>;
};

}  // namespace Tags
}  // namespace elliptic
