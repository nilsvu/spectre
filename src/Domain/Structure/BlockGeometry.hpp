// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace domain {

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Geometry of a block in the computational domain
 */
enum class BlockGeometry {
  /// A logical cube that can be deformed by coordinate maps
  Cube,
  /// A spherical shell
  SphericalShell
};

std::ostream& operator<<(std::ostream& os, BlockGeometry shell_type);

}  // namespace domain
