// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
/// [integrated_function]
double gaussian(double x, double mean, double factor) {
  return 2. * factor / sqrt(M_PI) * exp(-square(x - mean));
}

/// [integrated_function]
double integrable_singularity(double x, double factor) {
  return factor * cos(sqrt(abs(x))) / sqrt(abs(x));
}

SPECTRE_TEST_CASE("Unit.Numerical.Integration.GslQuadAdaptive",
                  "[Unit][NumericalAlgorithms]") {

{
  INFO("StandardGaussKronrod");
  // Construct the integration and give an example
  /// [integration_example]
  integrate::GslQuadAdaptive<integrate::IntegralType::StandardGaussKronrod,
                             double, double> integration{20};
  const double mean = 5.;
  const double factor = 2.0;
  const double lower_bound = -4.;
  const double upper_bound = 10.;
  integration.set_parameter<0>(mean);
  integration.set_parameter<1>(factor);
  integration.set_integrand(&gaussian);
  auto result = integration(lower_bound, upper_bound, 1.e-10, 4);
  /// [integration_example]
  CHECK(result.value == approx(factor * erf(upper_bound - mean) -
                                   factor * erf(lower_bound - mean)));
}

{
  INFO("InfiniteInterval");
  integrate::GslQuadAdaptive<integrate::IntegralType::InfiniteInterval, double,
                             double> integration{20};
  const double mean = 5.;
  const double factor = 2.0;
  integration.set_parameter<0>(mean);
  integration.set_parameter<1>(factor);
  integration.set_integrand(&gaussian);
  auto result = integration(1.e-10);
  CHECK(result.value == approx(2. * factor));
}

{
  INFO("UpperBoundaryInfinite");
  integrate::GslQuadAdaptive<integrate::IntegralType::UpperBoundaryInfinite,
                             double, double> integration{20};
  const double mean = 5.;
  const double factor = 2.0;
  const double lower_bound = -4.;
  integration.set_parameter<0>(mean);
  integration.set_parameter<1>(factor);
  integration.set_integrand(&gaussian);
  auto result = integration(lower_bound, 1.e-10);
  CHECK(result.value == approx(factor * (1 - erf(lower_bound - mean))));
}

{
  INFO("LowerBoundaryInfinite");
  integrate::GslQuadAdaptive<integrate::IntegralType::LowerBoundaryInfinite,
                             double, double> integration{20};
  const double mean = 5.;
  const double factor = 2.0;
  const double upper_bound = 10.;
  integration.set_parameter<0>(mean);
  integration.set_parameter<1>(factor);
  integration.set_integrand(&gaussian);
  auto result = integration(upper_bound, 1.e-10);
  CHECK(result.value == approx(factor * (1 + erf(upper_bound - mean))));
}

{
  INFO("IntegrableSingularitiesPresent");
  integrate::GslQuadAdaptive<
      integrate::IntegralType::IntegrableSingularitiesPresent, double>
      integration{20};
  const double mean = 5.;
  const double factor = 2.0;
  const double upper_bound = square(M_PI / 2.);
  integration.set_parameter<0>(factor);
  integration.set_integrand(&integrable_singularity);
  auto result = integration(0.0, upper_bound, 1.e-10);
  CHECK(result.value == approx(2. * factor * (sin(sqrt(upper_bound)))));
}

{
  INFO("IntegrableSingularitiesKnown");
  integrate::GslQuadAdaptive<
      integrate::IntegralType::IntegrableSingularitiesKnown, double>
      integration{30};
  const double mean = 5.;
  const double factor = 2.0;
  const double upper_bound = square(M_PI / 2.);
  const std::vector<double> points{-upper_bound, 0., upper_bound};
  integration.set_parameter<0>(factor);
  integration.set_integrand(&integrable_singularity);
  auto result = integration(points, 1.e-10);
  CHECK(result.value == approx(4. * factor * sin(sqrt(upper_bound))));
}

}
}  // namespace
