// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataVector.hpp"

namespace orbital {

template <typename DataType>
DataType tangential_velocity(double center_of_mass, double angular_velocity,
                             const DataType& position,
                             double eccentricity) noexcept;

struct ForceBalance {
  double eccentricity;
  DataVector position;
  DataVector conformal_factor_pow_4;
  DataVector lapse_square;
  DataVector shift_square;
  DataVector shift_y;
  DataVector dx_conformal_factor_pow_4;
  DataVector dx_lapse_square;
  DataVector dx_shift_square;
  DataVector dx_shift_y;

  std::array<double, 2> operator()(
      const std::array<double, 2>& center_of_mass_and_angular_velocity) const
      noexcept;
  std::array<std::array<double, 2>, 2> jacobian(
      const std::array<double, 2>& center_of_mass_and_angular_velocity) const
      noexcept;
};

}  // namespace orbital
