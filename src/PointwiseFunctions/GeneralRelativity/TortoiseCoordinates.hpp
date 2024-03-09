// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"

namespace gr {

DataVector boyer_lindquist_radius_minus_r_plus_from_tortoise(
    const DataVector& r_star, const double mass,
    const double dimensionless_spin);

}  // namespace gr
