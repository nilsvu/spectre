// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/BoundaryConditions.hpp"

#include <ostream>

#include "ErrorHandling/Error.hpp"

namespace elliptic {
std::ostream& operator<<(std::ostream& os,
                         const BoundaryCondition boundary_condition) noexcept {
  switch (boundary_condition) {
    case BoundaryCondition::Dirichlet:
      os << "Dirichlet";
      break;
    case BoundaryCondition::Neumann:
      os << "Neumann";
      break;
    default:
      ERROR("Invalid case");
  }
  return os;
}

}  // namespace elliptic
