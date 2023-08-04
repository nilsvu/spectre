// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
void test_test_helpers() {
  std::vector<double> vector{0, 1, 2, 3};
  test_iterators(vector);
  test_reverse_iterators(vector);

  std::set<double> set;
  set.insert(0);
  set.insert(1);
  set.insert(2);
  set.insert(3);
  test_iterators(set);
  test_reverse_iterators(set);

  std::unordered_set<int> u_set;
  u_set.insert(3);
  u_set.insert(2);
  u_set.insert(1);
  u_set.insert(0);
  test_iterators(u_set);

  Approx larger_approx =
      Approx::custom().epsilon(std::numeric_limits<double>::epsilon() * 1.e4);

  const std::vector<double> vec_a{1., 2., 3.5};
  CHECK_ITERABLE_APPROX(vec_a, vec_a);
  auto vec_b = vec_a;
  vec_b[1] += 1e-15;
  CHECK(vec_a != vec_b);
  CHECK_ITERABLE_APPROX(vec_a, vec_b);
  vec_b[1] += 1e-12;
  CHECK_ITERABLE_CUSTOM_APPROX(vec_a, vec_b, larger_approx);

  const std::vector<std::complex<double>> complex_vector{
      std::complex<double>(1.0, 1.5), std::complex<double>(2.0, 2.5),
      std::complex<double>(3.0, 3.5)};
  CHECK_ITERABLE_APPROX(complex_vector, complex_vector);
  auto perturbed_complex_vector = complex_vector;
  perturbed_complex_vector[1] += 1e-15;
  CHECK(complex_vector != perturbed_complex_vector);
  CHECK_ITERABLE_APPROX(complex_vector, perturbed_complex_vector);
  perturbed_complex_vector[1] += 1e-12;
  CHECK_ITERABLE_CUSTOM_APPROX(complex_vector, perturbed_complex_vector,
                               larger_approx);

  const std::vector<std::map<int, double>> vecmap_a{
      {{1, 1.}, {2, 2.}}, {{1, 1.23}, {3, 4.56}, {5, 7.89}}};
  CHECK_ITERABLE_APPROX(vecmap_a, vecmap_a);
  auto vecmap_b = vecmap_a;
  vecmap_b[1][1] += 1e-15;
  CHECK(vecmap_a != vecmap_b);
  CHECK_ITERABLE_APPROX(vecmap_a, vecmap_b);
  vecmap_b[1][1] += 1e-12;
  CHECK_ITERABLE_CUSTOM_APPROX(vecmap_a, vecmap_b, larger_approx);

  // Check that CHECK_ITERABLE_APPROX works on various containers
  {
    const std::set<int> a{1, 2, 3};
    CHECK_ITERABLE_APPROX(a, a);
  }
  {
    // Iteration order is unspecified, but we create the containers
    // differently so the order might differ between them.
    const std::unordered_set<int> a{1, 2, 3};
    const std::unordered_set<int> b{3, 2, 1};
    CHECK_ITERABLE_APPROX(a, b);
  }
  {
    // Iteration order is unspecified, but we create the containers
    // differently so the order might differ between them.
    const std::unordered_map<int, int> a{{1, 2}, {2, 3}, {3, 4}};
    const std::unordered_map<int, int> b{{3, 4}, {2, 3}, {1, 2}};
    CHECK_ITERABLE_APPROX(a, b);
  }
}

void test_derivative() {
  {  // 3D Test
    const std::array<double, 3> x{{1.2, -3.4, 1.3}};
    const double delta = 1.e-2;

    const auto func = [](const std::array<double, 3>& y) {
      return std::array<double, 3>{{sin(y[0]), cos(y[1]), exp(y[2])}};
    };
    const auto dfunc = [](const std::array<double, 3>& y) {
      return std::array<double, 3>{{cos(y[0]), -sin(y[1]), exp(y[2])}};
    };

    for (size_t i = 0; i < 3; ++i) {
      CHECK(gsl::at(numerical_derivative(func, x, i, delta), i) ==
            approx(gsl::at(dfunc(x), i)));
    }
  }
  {  // 2D Test
    const std::array<double, 2> x{{1.2, -2.4}};
    const double delta = 1.e-2;

    const auto func = [](const std::array<double, 2>& y) {
      return std::array<double, 2>{{sin(y[0]), cos(y[1])}};
    };
    const auto dfunc = [](const std::array<double, 2>& y) {
      return std::array<double, 2>{{cos(y[0]), -sin(y[1])}};
    };

    for (size_t i = 0; i < 2; ++i) {
      CHECK(gsl::at(numerical_derivative(func, x, i, delta), i) ==
            approx(gsl::at(dfunc(x), i)));
    }
  }
  {  // Non-array return type
    const std::array<double, 1> x{{1.2}};
    const double delta = 1.e-2;

    const auto func = [](const std::array<double, 1>& y) { return sin(y[0]); };
    const auto dfunc = [](const std::array<double, 1>& y) { return cos(y[0]); };

    CHECK(numerical_derivative(func, x, 0, delta) == approx(dfunc(x)));
  }
  {  // Floating point precision test
    const auto func = [](const std::array<double, 1>& /*x*/) { return 1.0; };
    CHECK(numerical_derivative(func, std::array{1.0}, 0, 1.e-2) == 0.0);
  }
}

void test_make_generator(const gsl::not_null<std::mt19937*> generator) {
  MAKE_GENERATOR(gen2);
  // This will fail randomly every 2**32 runs.  That is probably OK.
  CHECK((*generator)() != gen2());

  MAKE_GENERATOR(seeded_gen, 12345);
  CHECK(seeded_gen() == 3992670690);
}

void test_random_sample(const gsl::not_null<std::mt19937*> generator) {
  const std::vector<double> vec{1., 2., 3.5};
  for (const double rnd : random_sample<2>(vec, generator)) {
    CHECK((rnd == 1. or rnd == 2. or rnd == 3.5));
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(random_sample<5>(vec, generator),
                    Catch::Matchers::ContainsSubstring(
                        "Cannot take 5 samples from container of size 3"));
#endif
  const std::unordered_set ints{1, 4, 2, 3};
  const std::vector<int> two_samples = random_sample(2, ints, generator);
  CHECK(two_samples.size() == 2);
  const std::vector<int> over_sampled = random_sample(10, ints, generator);
  CHECK(over_sampled.size() == ints.size());
  const auto check_sample = [&ints](const std::vector<int>& samples) {
    for (const auto& sample : samples) {
      CHECK(alg::count(ints, sample) == 1);
      CHECK(alg::count(samples, sample) == 1);
    }
  };
  check_sample(two_samples);
  check_sample(over_sampled);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.TestHelpers", "[Unit]") {
  test_test_helpers();
  test_derivative();
  MAKE_GENERATOR(gen1);
  test_make_generator(make_not_null(&gen1));
  test_random_sample(make_not_null(&gen1));
}
