// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct LorentzianProxy : Poisson::Solutions::Lorentzian<Dim> {
  using Poisson::Solutions::Lorentzian<Dim>::Lorentzian;

  using field_tags = tmpl::list<
      Poisson::Tags::Field,
      ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>;
  using source_tags = tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Lorentzian<Dim>::variables(x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Lorentzian<Dim>::variables(x, source_tags{});
  }
};

template <size_t Dim, typename... Maps>
void test_solution(const Mesh<Dim>& mesh,
                   const domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                               Maps...>& coord_map,
                   const double tolerance) {
  const LorentzianProxy<Dim> solution{};
  pypp::check_with_random_values<
      1, tmpl::list<Poisson::Tags::Field,
                    ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>,
                                  Frame::Inertial>>>(
      &LorentzianProxy<Dim>::field_variables, solution, "Lorentzian",
      {"field", "field_gradient"}, {{{-5., 5.}}}, std::make_tuple(),
      DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>>(
      &LorentzianProxy<Dim>::source_variables, solution, "Lorentzian",
      {"source"}, {{{-5., 5.}}}, std::make_tuple(), DataVector(5));

  const Poisson::Solutions::Lorentzian<Dim> check_solution{};
  const Poisson::Solutions::Lorentzian<Dim> created_solution =
      test_creation<Poisson::Solutions::Lorentzian<Dim>>("  ");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);

  FirstOrderEllipticSolutionsTestHelpers::verify_solution<
      Poisson::FirstOrderSystem<Dim>>(
      created_solution, typename Poisson::FirstOrderSystem<Dim>::fluxes{}, mesh,
      coord_map, tolerance);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.Lorentzian",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};
  // 1D and 2D solutions are not implemented yet.
  using AffineMap = domain::CoordinateMaps::Affine;
  using AffineMap3D =
      domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
  test_solution(
      Mesh<3>{12, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>{
          {{-1., 1., -0.5, 0.5}, {-1., 1., -0.5, 0.5}, {-1., 1., -0.5, 0.5}}},
      1.e-1);
}
