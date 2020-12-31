// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::Tags {

template <typename Tag>
struct BoundaryConditionType : db::PrefixTag {
  using type = elliptic::BoundaryConditionType;
};

template <typename Tags>
struct BoundaryConditionTypes : db::SimpleTag {
  using type = tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<elliptic::Tags::BoundaryCondition, Tags>>;
};

}  // namespace elliptic::Tags
