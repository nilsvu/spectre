// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace elliptic {

enum class BoundaryCondition { Dirichlet, Neumann };

namespace Tags {

struct BoundaryCondition : db::SimpleTag {
  using type = ::elliptic::BoundaryCondition;
};

}  // namespace Tags

}  // namespace elliptic
