// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Protocols.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "tests/Unit/ProtocolTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct ProductOfSinusoidsProxy : Poisson::Solutions::ProductOfSinusoids<Dim> {
  using base = Poisson::Solutions::ProductOfSinusoids<Dim>;
  using base::ProductOfSinusoids;
  using supported_tags = typename base::supported_tags;

  tuples::tagged_tuple_from_typelist<supported_tags> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return base::variables(x, supported_tags{});
  }
};

template <size_t Dim>
void test_solution(const std::array<double, Dim>& wave_numbers,
                   const std::string& options) {
  const ProductOfSinusoidsProxy<Dim> solution{wave_numbers};
  pypp::check_with_random_values<
      1, typename Poisson::Solutions::ProductOfSinusoids<Dim>::supported_tags>(
      &ProductOfSinusoidsProxy<Dim>::variables, solution, "ProductOfSinusoids",
      {"field", "field_gradient", "source"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers), DataVector(5));

  Poisson::Solutions::ProductOfSinusoids<Dim> created_solution =
      TestHelpers::test_creation<Poisson::Solutions::ProductOfSinusoids<Dim>>(
          "WaveNumbers: " + options);
  CHECK(created_solution == solution);
  test_serialization(solution);
}

}  // namespace

test_protocol_conformance<Poisson::Solutions::ProductOfSinusoids<1>,
                          elliptic::protocols::AnalyticSolution>;
test_protocol_conformance<Poisson::Solutions::ProductOfSinusoids<2>,
                          elliptic::protocols::AnalyticSolution>;
test_protocol_conformance<Poisson::Solutions::ProductOfSinusoids<3>,
                          elliptic::protocols::AnalyticSolution>;

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.ProductOfSinusoids",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};

  using AffineMap = domain::CoordinateMaps::Affine;
  {
    INFO("1D");
    test_solution<1>({{0.5}}, "[0.5]");

    using system = Poisson::FirstOrderSystem<1>;
    const Poisson::Solutions::ProductOfSinusoids<1> solution{{{0.5}}};
    const typename system::fluxes fluxes_computer{};
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap>
        coord_map{{-1., 1., 0., M_PI}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, fluxes_computer, coord_map, 1.e5, 3.);
  }
  {
    INFO("2D");
    test_solution<2>({{0.5, 1.}}, "[0.5, 1.]");

    using system = Poisson::FirstOrderSystem<2>;
    const Poisson::Solutions::ProductOfSinusoids<2> solution{{{0.5, 0.5}}};
    const typename system::fluxes fluxes_computer{};
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>
        coord_map{{{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, fluxes_computer, coord_map, 1.e5, 3.);
  }
  {
    INFO("3D");
    test_solution<3>({{1., 0.5, 1.5}}, "[1., 0.5, 1.5]");

    using system = Poisson::FirstOrderSystem<3>;
    const Poisson::Solutions::ProductOfSinusoids<3> solution{{{0.5, 0.5, 0.5}}};
    const typename system::fluxes fluxes_computer{};
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{
            {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, fluxes_computer, coord_map, 1.e5, 3.);
  }
}
