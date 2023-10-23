// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE("Unit.Spectral.SphericalHarmonics",
                  "[NumericalAlgorithms][Unit]") {
  const size_t num_points = 6;
  {
    INFO("Theta");
    // Theta in [0, pi] are Gauss-Legendre points in cos(theta)
    const auto xi =
        Spectral::collocation_points<Spectral::Basis::SphericalHarmonic,
                                     Spectral::Quadrature::Gauss>(num_points);
    auto legendre_gauss_points =
        Spectral::collocation_points<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::Gauss>(num_points);
    CHECK_ITERABLE_APPROX(xi, legendre_gauss_points);
  }
  {
    INFO("Phi");
    // Phi in [0, 2 pi] are equidistant points in phi
    const auto eta =
        Spectral::collocation_points<Spectral::Basis::SphericalHarmonic,
                                     Spectral::Quadrature::Equiangular>(
            num_points);
    auto equidistant_points =
        Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::CellCentered>(
            num_points);
    CHECK_ITERABLE_APPROX(eta, equidistant_points);
  }
}
