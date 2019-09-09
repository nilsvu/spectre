// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <pup.h>
#include <random>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

void test_solution() noexcept {
  const auto star = test_creation<Xcts::Solutions::TovStar>(
      "  CentralDensity: 1.0e-5\n"
      "  PolytropicConstant: 0.001\n"
      "  PolytropicExponent: 1.4");
  CHECK(star == Xcts::Solutions::TovStar(0.00001, 0.001, 1.4));
  test_serialization(star);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.AnalyticSolutions.Xcts.TovStar",
                  "[Unit][PointwiseFunctions]") {
  // test_solution();

  Xcts::Solutions::TovStar solution(
      0.0008087415253997405,  // Central enthalpy h=1.2
      123.6489, 2.);
  // Xcts::Solutions::TovStar solution(1.e-3, 8., 2.);
  const auto& radial_solution = solution.radial_solution();
  const double isotropic_radius_of_star = radial_solution.outer_radius();
  // const double mass_of_star = radial_solution.mass(radius_of_star);
  Approx numerical_approx = Approx::custom().epsilon(1e-2).scale(1.);
  CHECK(isotropic_radius_of_star == numerical_approx(9.749166076324654));
  // CHECK(mass_of_star == numerical_approx(1.926902861434354));

  CHECK(radial_solution.conformal_factor(isotropic_radius_of_star) ==
        numerical_approx(1.0783609635876443));

  const auto test_vars = solution.variables(
      tnsr::I<DataVector, 3>{
          {{{0.,          0.2020202,   0.4040404,   0.60606061,  0.80808081,
             1.01010101,  1.21212121,  1.41414141,  1.61616162,  1.81818182,
             2.02020202,  2.22222222,  2.42424242,  2.62626263,  2.82828283,
             3.03030303,  3.23232323,  3.43434343,  3.63636364,  3.83838384,
             4.04040404,  4.24242424,  4.44444444,  4.64646465,  4.84848485,
             5.05050505,  5.25252525,  5.45454545,  5.65656566,  5.85858586,
             6.06060606,  6.26262626,  6.46464646,  6.66666667,  6.86868687,
             7.07070707,  7.27272727,  7.47474747,  7.67676768,  7.87878788,
             8.08080808,  8.28282828,  8.48484848,  8.68686869,  8.88888889,
             9.09090909,  9.29292929,  9.49494949,  9.6969697,   9.8989899,
             10.1010101,  10.3030303,  10.50505051, 10.70707071, 10.90909091,
             11.11111111, 11.31313131, 11.51515152, 11.71717172, 11.91919192,
             12.12121212, 12.32323232, 12.52525253, 12.72727273, 12.92929293,
             13.13131313, 13.33333333, 13.53535354, 13.73737374, 13.93939394,
             14.14141414, 14.34343434, 14.54545455, 14.74747475, 14.94949495,
             15.15151515, 15.35353535, 15.55555556, 15.75757576, 15.95959596,
             16.16161616, 16.36363636, 16.56565657, 16.76767677, 16.96969697,
             17.17171717, 17.37373737, 17.57575758, 17.77777778, 17.97979798,
             18.18181818, 18.38383838, 18.58585859, 18.78787879, 18.98989899,
             19.19191919, 19.39393939, 19.5959596,  19.7979798,  20.},
            {100, 0.},
            {100, 0.}}}},
      tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>{});
  // const DataVector expected_conformal_factor{
  //     1.16891041, 1.16790224, 1.16493608, 1.16007134, 1.15341765, 1.14515032,
  //     1.13552894, 1.12491342, 1.1137671,  1.10263415,
  //     1.09208386, 1.08263028};
  // CHECK_ITERABLE_APPROX(
  //     get(get<Xcts::Tags::ConformalFactor<DataVector>>(test_vars)),
  //     expected_conformal_factor);
  CAPTURE(get(get<Xcts::Tags::ConformalFactor<DataVector>>(test_vars)));
  CHECK(false);
}
