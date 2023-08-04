// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {
namespace {
using Wedge2D = CoordinateMaps::Wedge<2>;

void test_wedge2d_all_orientations(const bool with_equiangular_map) {
  INFO("Wedge2d all orientations");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> unit_dis(0, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(4, 7);

  // Check that points on the corners of the reference square map to the correct
  // corners of the wedge.
  const std::array<double, 2> lower_right_corner{{1.0, -1.0}};
  const std::array<double, 2> upper_right_corner{{1.0, 1.0}};
  CAPTURE(gsl::at(lower_right_corner, 0));
  CAPTURE(gsl::at(upper_right_corner, 0));
  CAPTURE(gsl::at(lower_right_corner, 1));
  CAPTURE(gsl::at(upper_right_corner, 1));

  const double random_inner_radius_upper_xi = inner_dis(gen);
  CAPTURE(random_inner_radius_upper_xi);
  const double random_inner_radius_upper_eta = inner_dis(gen);
  CAPTURE(random_inner_radius_upper_eta);
  const double random_inner_radius_lower_xi = inner_dis(gen);
  CAPTURE(random_inner_radius_lower_xi);
  const double random_inner_radius_lower_eta = inner_dis(gen);
  CAPTURE(random_inner_radius_lower_eta);
  const double random_outer_radius_upper_xi = outer_dis(gen);
  CAPTURE(random_outer_radius_upper_xi);
  const double random_outer_radius_upper_eta = outer_dis(gen);
  CAPTURE(random_outer_radius_upper_eta);
  const double random_outer_radius_lower_xi = outer_dis(gen);
  CAPTURE(random_outer_radius_lower_xi);
  const double random_outer_radius_lower_eta = outer_dis(gen);
  CAPTURE(random_outer_radius_lower_eta);

  const Wedge2D map_upper_xi(
      random_inner_radius_upper_xi, random_outer_radius_upper_xi, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
      with_equiangular_map);
  const Wedge2D map_upper_eta(
      random_inner_radius_upper_eta, random_outer_radius_upper_eta, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
      with_equiangular_map);
  const Wedge2D map_lower_xi(
      random_inner_radius_lower_xi, random_outer_radius_lower_xi, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
      with_equiangular_map);
  const Wedge2D map_lower_eta(
      random_inner_radius_lower_eta, random_outer_radius_lower_eta, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
      with_equiangular_map);
  CHECK(map_lower_eta != map_lower_xi);
  CHECK(map_upper_eta != map_lower_eta);
  CHECK(map_lower_eta != map_upper_xi);

  CHECK(map_upper_xi(lower_right_corner)[0] ==
        approx(random_outer_radius_upper_xi / sqrt(2.0)));
  CHECK(map_upper_eta(lower_right_corner)[1] ==
        approx(-random_outer_radius_upper_eta / sqrt(2.0)));
  CHECK(map_lower_xi(upper_right_corner)[0] ==
        approx(-random_outer_radius_lower_xi / sqrt(2.0)));
  CHECK(map_lower_eta(upper_right_corner)[1] ==
        approx(random_outer_radius_lower_eta / sqrt(2.0)));

  // Check that random points on the edges of the reference square map to the
  // correct edges of the wedge.
  const std::array<double, 2> random_right_edge{{1.0, real_dis(gen)}};
  const std::array<double, 2> random_left_edge{{-1.0, real_dis(gen)}};

  CHECK(magnitude(map_upper_xi(random_right_edge)) ==
        approx(random_outer_radius_upper_xi));
  CHECK(map_upper_xi(random_left_edge)[0] ==
        approx(random_inner_radius_upper_xi / sqrt(2.0)));
  CHECK(magnitude(map_upper_eta(random_right_edge)) ==
        approx(random_outer_radius_upper_eta));
  CHECK(map_upper_eta(random_left_edge)[1] ==
        approx(-random_inner_radius_upper_eta / sqrt(2.0)));
  CHECK(magnitude(map_lower_xi(random_right_edge)) ==
        approx(random_outer_radius_lower_xi));
  CHECK(map_lower_xi(random_left_edge)[0] ==
        approx(-random_inner_radius_lower_xi / sqrt(2.0)));
  CHECK(magnitude(map_lower_eta(random_right_edge)) ==
        approx(random_outer_radius_lower_eta));
  CHECK(map_lower_eta(random_left_edge)[1] ==
        approx(random_inner_radius_lower_eta / sqrt(2.0)));

  const double inner_radius = inner_dis(gen);
  CAPTURE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE(outer_radius);
  const double inner_circularity = unit_dis(gen);
  CAPTURE(inner_circularity);
  const double outer_circularity = unit_dis(gen);
  CAPTURE(outer_circularity);

  using WedgeHalves = Wedge2D::WedgeHalves;
  const std::array<WedgeHalves, 3> possible_halves = {
      {WedgeHalves::UpperOnly, WedgeHalves::LowerOnly, WedgeHalves::Both}};
  for (OrientationMapIterator<2> map_i{}; map_i; ++map_i) {
    if (get(determinant(discrete_rotation_jacobian(*map_i))) < 0.0) {
      continue;
    }
    const auto& orientation = map_i();
    CAPTURE(orientation);
    for (const auto& halves : possible_halves) {
      CAPTURE(halves);
      for (const auto radial_distribution :
           {CoordinateMaps::Distribution::Linear,
            CoordinateMaps::Distribution::Logarithmic,
            CoordinateMaps::Distribution::Inverse}) {
        CAPTURE(radial_distribution);
        test_suite_for_map_on_unit_cube(Wedge2D{
            inner_radius, outer_radius,
            radial_distribution == CoordinateMaps::Distribution::Linear
                ? inner_circularity
                : 1.0,
            radial_distribution == CoordinateMaps::Distribution::Linear
                ? outer_circularity
                : 1.0,
            orientation, with_equiangular_map, halves, radial_distribution});
      }
    }
  }
}

void test_wedge2d_fail() {
  INFO("Wedge2d fail");
  const auto map = Wedge2D(0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);

  // Any point with x<=0 should fail the inverse map.
  const std::array<double, 2> test_mapped_point1{{0.0, 3.0}};
  const std::array<double, 2> test_mapped_point2{{0.0, -6.0}};
  const std::array<double, 2> test_mapped_point3{{-1.0, 3.0}};

  // This point is outside the mapped wedge.  So inverse should either
  // return the correct inverse (which happens to be computable for
  // this point) or it should return nullopt.
  const std::array<double, 2> test_mapped_point4{{100.0, -6.0}};

  CHECK_FALSE(map.inverse(test_mapped_point1).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point2).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point3).has_value());
  if (map.inverse(test_mapped_point4).has_value()) {
    CHECK_ITERABLE_APPROX(map(map.inverse(test_mapped_point4).value()),
                          test_mapped_point4);
  }
}

void test_equality() {
  INFO("Equality");
  const auto wedge2d = Wedge2D(0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_inner_radius_changed =
      Wedge2D(0.3, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_outer_radius_changed =
      Wedge2D(0.2, 4.2, 0.0, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_inner_circularity_changed =
      Wedge2D(0.2, 4.0, 0.3, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_outer_circularity_changed =
      Wedge2D(0.2, 4.0, 0.0, 0.9, OrientationMap<2>{}, true);
  const auto wedge2d_orientation_map_changed =
      Wedge2D(0.2, 4.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
              true);
  const auto wedge2d_use_equiangular_map_changed =
      Wedge2D(0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, false);
  CHECK_FALSE(wedge2d == wedge2d_inner_radius_changed);
  CHECK_FALSE(wedge2d == wedge2d_outer_radius_changed);
  CHECK_FALSE(wedge2d == wedge2d_inner_circularity_changed);
  CHECK_FALSE(wedge2d == wedge2d_outer_circularity_changed);
  CHECK_FALSE(wedge2d == wedge2d_orientation_map_changed);
  CHECK_FALSE(wedge2d == wedge2d_use_equiangular_map_changed);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D.Map", "[Domain][Unit]") {
  test_wedge2d_fail();
  test_wedge2d_all_orientations(false);  // Equidistant
  test_wedge2d_all_orientations(true);   // Equiangular
  test_equality();
  CHECK(not Wedge2D{}.is_identity());

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      Wedge2D(-0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true),
      Catch::Matchers::ContainsSubstring(
          "The radius of the inner surface must be greater than zero."));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, 4.0, -0.2, 1.0, OrientationMap<2>{}, true),
      Catch::Matchers::ContainsSubstring(
          "Sphericity of the inner surface must be between 0 and 1"));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, 4.0, 0.0, -0.2, OrientationMap<2>{}, true),
      Catch::Matchers::ContainsSubstring(
          "Sphericity of the outer surface must be between 0 and 1"));
  CHECK_THROWS_WITH(Wedge2D(4.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true),
                    Catch::Matchers::ContainsSubstring(
                        "The radius of the outer surface must be greater than "
                        "the radius of the inner surface."));
  CHECK_THROWS_WITH(
      Wedge2D(3.0, 4.0, 1.0, 0.0, OrientationMap<2>{}, true),
      Catch::Matchers::ContainsSubstring(
          "The arguments passed into the constructor for Wedge result in an "
          "object where the outer surface is pierced by the inner surface."));
#endif
}
}  // namespace domain
