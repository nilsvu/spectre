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
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct ProductOfSinusoidsProxy : Poisson::Solutions::ProductOfSinusoids<Dim> {
  using Poisson::Solutions::ProductOfSinusoids<Dim>::ProductOfSinusoids;

  using field_tags = tmpl::list<
      Poisson::Tags::Field,
      ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>;
  using source_tags = tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::ProductOfSinusoids<Dim>::variables(x,
                                                                  field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::ProductOfSinusoids<Dim>::variables(
        x, source_tags{});
  }
};

template <size_t Dim, typename... Maps>
void test_solution(const std::array<double, Dim>& wave_numbers,
                   const std::string& options, const Mesh<Dim>& mesh,
                   const domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                               Maps...>& coord_map,
                   const double tolerance) {
  const ProductOfSinusoidsProxy<Dim> solution(wave_numbers);
  pypp::check_with_random_values<
      1, tmpl::list<Poisson::Tags::Field,
                    ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>,
                                  Frame::Inertial>>>(
      &ProductOfSinusoidsProxy<Dim>::field_variables, solution,
      "ProductOfSinusoids", {"field", "field_gradient"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers), DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>>(
      &ProductOfSinusoidsProxy<Dim>::source_variables, solution,
      "ProductOfSinusoids", {"source"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers), DataVector(5));

  Poisson::Solutions::ProductOfSinusoids<Dim> created_solution =
      test_creation<Poisson::Solutions::ProductOfSinusoids<Dim>>(
          "  WaveNumbers: " + options);
  CHECK(created_solution == solution);
  test_serialization(solution);

  FirstOrderEllipticSolutionsTestHelpers::verify_solution<
      Poisson::FirstOrderSystem<Dim>>(
      created_solution, typename Poisson::FirstOrderSystem<Dim>::fluxes{}, mesh,
      coord_map, tolerance);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.ProductOfSinusoids",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};

  using AffineMap = domain::CoordinateMaps::Affine;
  {
    INFO("1D");
    test_solution<1>(
        {{0.5}}, "[0.5]",
        Mesh<1>{10, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap>{
            {-1., 1., 0., M_PI}},
        1.e-8);
  }
  {
    INFO("2D");
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    test_solution<2>(
        {{0.5, 1.}}, "[0.5, 1.]",
        Mesh<2>{10, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>{
            {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}},
        1.e-5);
  }
  {
    INFO("3D");
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    test_solution<3>(
        {{1., 0.5, 1.5}}, "[1., 0.5, 1.5]",
        Mesh<3>{10, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>{
            {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}},
        1.e-3);
  }
}
