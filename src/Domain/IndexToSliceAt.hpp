// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Side.hpp"

/// \ingroup ComputationalDomainGroup
/// Finds the index in the perpendicular dimension of an element boundary
template <size_t Dim>
size_t index_to_slice_at(const Index<Dim>& extents,
                         const Direction<Dim>& direction,
                         const size_t offset = 0) noexcept {
  return direction.side() == Side::Lower
             ? offset
             : extents[direction.dimension()] - 1 - offset;
}
