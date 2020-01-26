// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Math.hpp"

namespace LinearSolver {
namespace schwarz_detail {

template <typename DataType>
DataType smoothstep(const DataType& arg) noexcept {
  static const std::vector<double> coeffs{0., 15., 0., -10., 0., 3.};
  DataType result = make_with_value<DataType>(arg, 0.);
  for (size_t i = 0; i < get_size(arg); i++) {
    get_element(result, i) =
        get_element(result, i) > 1.
            ? 1.
            : get_element(result, i) < -1.
                  ? -1.
                  : 0.125 * evaluate_polynomial(coeffs, get_element(arg, i));
  }
  return result;
}

template <typename DataType>
DataType weight(const DataType& logical_coord, const double width) noexcept {
  return 0.5 * (smoothstep((logical_coord + 1) / width) -
                smoothstep((logical_coord - 1) / width));
}

}  // namespace schwarz_detail
}  // namespace LinearSolver
