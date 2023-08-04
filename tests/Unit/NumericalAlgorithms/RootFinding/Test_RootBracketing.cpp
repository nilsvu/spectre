// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <optional>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/RootBracketing.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
std::optional<double> f_free(double x) {
  return (x < 1.0 or x > 2.0) ? std::nullopt
                              : std::optional<double>(2.0 - square(x));
}
// f_root_near_bounds is a case contrived to generate an error
// because the root is within 1e-10 of the bounds.
std::optional<double> f_root_near_bounds(double x) {
  return (x < 1.0 or x > 2.0) ? std::nullopt
                              : std::optional<double>(x-1.0-1.e-10);
}
struct F {
  std::optional<double> operator()(double x) const {
    return (x < 1.0 or x > 2.0) ? std::nullopt
                                : std::optional<double>(2.0 - square(x));
  }
};

template <typename Function>
void test_bracketing_simple_one_function(const Function& f,
                                         const std::array<double, 2>& bounds,
                                         const std::optional<double>& guess) {
  double lower = bounds[0];
  double upper = bounds[1];
  double f_at_lower = std::numeric_limits<double>::signaling_NaN();
  double f_at_upper = std::numeric_limits<double>::signaling_NaN();
  if (guess.has_value()) {
    RootFinder::bracket_possibly_undefined_function_in_interval(
        &lower, &upper, &f_at_lower, &f_at_upper, f, guess.value());
  } else {
    RootFinder::bracket_possibly_undefined_function_in_interval(
        &lower, &upper, &f_at_lower, &f_at_upper, f);
  }
  CHECK(f_at_lower * f_at_upper <= 0.0);
  CHECK(lower <= sqrt(2.0));
  CHECK(upper >= sqrt(2.0));
}

void test_bracketing_simple_multiple_functions(
    const std::array<double, 2>& bounds, const std::optional<double>& guess) {
  const auto f_lambda = [](double x) -> std::optional<double> {
    return (x < 1.0 or x > 2.0) ? std::nullopt
                                : std::optional<double>(2.0 - square(x));
  };
  const F f_functor{};

  test_bracketing_simple_one_function(f_free, bounds, guess);
  test_bracketing_simple_one_function(f_lambda, bounds, guess);
  test_bracketing_simple_one_function(f_functor, bounds, guess);
}

void test_bracketing_simple() {
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);

  // Set up random upper/lower bounds.
  const std::array<double, 2> bounds_invalid{
      {unit_dis(gen), 2.0 + unit_dis(gen)}};
  // Lower bound is less than sqrt(2.0) but valid.
  const std::array<double, 2> bounds_upper_is_invalid{
      {1.0 + (sqrt(2.0) - 1.0 - std::numeric_limits<double>::epsilon()) *
                 unit_dis(gen),
       2.0 + unit_dis(gen)}};
  // Upper bound is greater than sqrt(2.0) but valid.
  const std::array<double, 2> bounds_lower_is_invalid{
      {unit_dis(gen),
       sqrt(2.0) + (2.0 - sqrt(2.0) + std::numeric_limits<double>::epsilon()) *
                       unit_dis(gen)}};
  // Both bounds bracket root but are valid.
  const std::array<double, 2> bounds_valid{
      {1.0 + (sqrt(2.0) - 1.0 - std::numeric_limits<double>::epsilon()) *
                 unit_dis(gen),
       sqrt(2.0) + (2.0 - sqrt(2.0) + std::numeric_limits<double>::epsilon()) *
                       unit_dis(gen)}};

  const auto test_with_and_without_guess =
      [&gen, &unit_dis](const std::array<double, 2>& bounds) {
        test_bracketing_simple_multiple_functions(bounds, std::nullopt);
        test_bracketing_simple_multiple_functions(
            bounds, bounds[0] + unit_dis(gen) * (bounds[1] - bounds[0]));
      };

  test_with_and_without_guess(bounds_invalid);
  test_with_and_without_guess(bounds_upper_is_invalid);
  test_with_and_without_guess(bounds_lower_is_invalid);
  test_with_and_without_guess(bounds_valid);
}

void test_bracketing_datavector() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);

  const DataVector lower{
      unit_dis(gen),
      1.0 + (sqrt(2.0) - 1.0 - std::numeric_limits<double>::epsilon()) *
                unit_dis(gen),
      unit_dis(gen),
      1.0 + (sqrt(2.0) - 1.0 - std::numeric_limits<double>::epsilon()) *
                unit_dis(gen)};
  const DataVector upper{
      2.0 + unit_dis(gen), 2.0 + unit_dis(gen),
      sqrt(2.0) + (2.0 - sqrt(2.0) + std::numeric_limits<double>::epsilon()) *
                      unit_dis(gen),
      sqrt(2.0) + (2.0 - sqrt(2.0) + std::numeric_limits<double>::epsilon()) *
                      unit_dis(gen)};

  const auto f_lambda = [](double x, size_t /*i*/) -> std::optional<double> {
    return (x < 1.0 or x > 2.0) ? std::nullopt
                                : std::optional<double>(2.0 - square(x));
  };

  const auto do_test = [&f_lambda](DataVector lower_l, DataVector upper_l,
                                   const std::optional<DataVector>& guess) {
    DataVector f_at_lower(lower_l.size(),
                          std::numeric_limits<double>::signaling_NaN());
    DataVector f_at_upper(upper_l.size(),
                          std::numeric_limits<double>::signaling_NaN());
    if (guess.has_value()) {
      RootFinder::bracket_possibly_undefined_function_in_interval(
          &lower_l, &upper_l, &f_at_lower, &f_at_upper, f_lambda,
          guess.value());
    } else {
      RootFinder::bracket_possibly_undefined_function_in_interval(
          &lower_l, &upper_l, &f_at_lower, &f_at_upper, f_lambda);
    }
    for (size_t s = 0; s < f_at_lower.size(); ++s) {
      CHECK(f_at_lower[s] * f_at_upper[s] <= 0.0);
      CHECK(lower_l[s] <= sqrt(2.0));
      CHECK(upper_l[s] >= sqrt(2.0));
    }
  };

  do_test(lower, upper, std::nullopt);
  do_test(lower, upper, DataVector(lower + (upper - lower) * unit_dis(gen)));
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.Bracketing",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  test_bracketing_simple();
  test_bracketing_datavector();

  CHECK_THROWS_WITH((test_bracketing_simple_one_function(
                        f_root_near_bounds, {{0.1, 1.1}}, std::nullopt)),
                    Catch::Matchers::ContainsSubstring(
                        "bracket_by_contracting: Cannot bracket root between"));
}
}  // namespace
