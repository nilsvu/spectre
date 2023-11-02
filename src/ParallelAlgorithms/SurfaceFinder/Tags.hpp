// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"

namespace SurfaceFinder::Tags {

template <typename TemporalIdType>
struct FilledRadii : db::SimpleTag {
  using type = std::unordered_map<TemporalIdType,
                                  std::pair<DataVector, std::vector<bool>>>;
};

}  // namespace SurfaceFinder::Tags
