// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"

#include <cmath>
#include <korb.hpp>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace gr {

DataVector boyer_lindquist_radius_minus_r_plus_from_tortoise(
    const DataVector& r_star, const double mass,
    const double dimensionless_spin) {
  DataVector r{r_star.size()};
  for (size_t i = 0; i < r_star.size(); i++) {
    r[i] = korb_rsubtrplusfromrs(r_star[i], mass * dimensionless_spin);
  }
  return r;
}

}  // namespace gr
