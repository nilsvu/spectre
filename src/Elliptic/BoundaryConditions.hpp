// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

#include "DataStructures/DataBox/Tag.hpp"

namespace elliptic {

enum class BoundaryCondition { Dirichlet, Neumann };

std::ostream& operator<<(std::ostream& os,
                         const BoundaryCondition boundary_condition) noexcept;

namespace Tags {

struct BoundaryCondition : db::SimpleTag {
  using type = ::elliptic::BoundaryCondition;
};

}  // namespace Tags

}  // namespace elliptic
