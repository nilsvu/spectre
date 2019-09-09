// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <class F>
void test_cubic_spline(const F& function, const double lower_bound,
                       const double upper_bound, const size_t size,
                       const double tolerance) noexcept {
  // Construct random points between lower and upper bound to interpolate
  // through. Always include the bounds in the x-values.
  std::vector<double> x_values(size), y_values(size);
  const double delta_x = (upper_bound - lower_bound) / size;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, delta_x);
  x_values.front() = lower_bound;
  for (size_t i = 1; i < size - 1; ++i) {
    x_values[i] = lower_bound + i * delta_x + dist(gen);
  }
  x_values.back() = upper_bound;
  for (size_t i = 0; i < size; ++i) {
    y_values[i] = function(x_values[i]);
  }

  Approx custom_approx = Approx::custom().epsilon(tolerance).scale(1.0);

  // Construct the interpolant and give an example
  /// [interpolate_example]
  intrp::CubicSpline interpolant{x_values, y_values};
  const double x_to_interpolate_to = (upper_bound - lower_bound) / 2.;
  CHECK(interpolant(x_to_interpolate_to) ==
        custom_approx(function(x_to_interpolate_to)));
  /// [interpolate_example]

  // Check that the interpolation matches the function within the given
  // tolerance. Also check that the serialized-and-deserialized interpolant does
  // the same.
  const auto deserialized_interpolant = serialize_and_deserialize(interpolant);
  for (size_t i = 0; i < 10 * size; ++i) {
    const double x_value = lower_bound + i * delta_x * 0.1 + 0.1 * dist(gen);
    CAPTURE(x_value);
    const double y_value = function(x_value);
    CHECK(interpolant(x_value) == custom_approx(y_value));
    CHECK(deserialized_interpolant(x_value) == custom_approx(y_value));
  }

  // Make sure moving the interpolant doesn't break anything
  const auto moved_interpolant = std::move(interpolant);
  const double x_value = lower_bound + dist(gen) * size;
  const double y_value = function(x_value);
  CHECK(moved_interpolant(x_value) == custom_approx(y_value));
}

void test_with_polynomial(const size_t number_of_points, const size_t degree,
                          const double tolerance) {
  INFO("Polynomial degree := " << degree);
  std::vector<double> coeffs(degree, 1.);
  test_cubic_spline(
      [&coeffs](const auto& x) noexcept {
        return evaluate_polynomial(coeffs, x);
      },
      -1.0, 2.3, number_of_points, tolerance);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.CubicSpline",
                  "[Unit][NumericalAlgorithms]") {
  test_with_polynomial(10, 1, 1.e-12);
  test_with_polynomial(10, 2, 1.e-12);
  test_with_polynomial(100, 3, 1.e-3);
  test_with_polynomial(1000, 3, 1.e-3);
}
