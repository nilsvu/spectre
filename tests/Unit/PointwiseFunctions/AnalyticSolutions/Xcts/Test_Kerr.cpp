// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace {

void test_solution(const double mass, const std::array<double, 3> spin,
                   const std::array<double, 3> center,
                   const std::string& options_string) {
  CAPTURE(mass);
  CAPTURE(spin);
  CAPTURE(center);
  const auto created =
      TestHelpers::test_factory_creation<Xcts::Solutions::AnalyticSolution<
          tmpl::list<Xcts::Solutions::Registrars::Kerr>>>(options_string);
  REQUIRE(dynamic_cast<const Kerr<>*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const Kerr<>&>(*created);
  {
    INFO("Properties");
    CHECK(solution.mass() == mass);
    CHECK(solution.dimensionless_spin() == spin);
    CHECK(solution.center() == center);
  }
  // {
  //   INFO("Semantics");
  //   test_serialization(solution);
  //   test_copy_semantics(solution);
  //   auto move_solution = solution;
  //   test_move_semantics(std::move(move_solution), solution);
  // }
  {
    INFO("Verify the solution solves the XCTS system");
    const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const double inner_radius = 2.;
    const double outer_radius = 6.;
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{{{-1., 1., inner_radius, outer_radius},
                   {-1., 1., inner_radius, outer_radius},
                   {-1., 1., inner_radius, outer_radius}}};
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = coord_map(logical_coords);
    const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
    const auto get_items = [](const auto&... args) {
      return std::forward_as_tuple(args...);
    };
    const auto& analytic_data =
        dynamic_cast<const Xcts::AnalyticData::AnalyticData<
            tmpl::list<Xcts::Solutions::Registrars::Kerr>>&>(*created);
    {
      INFO("Hamiltonian equation");
      using system = Xcts::FirstOrderSystem<Xcts::Equations::Hamiltonian,
                                            Xcts::Geometry::Curved>;
      const auto background_fields =
          analytic_data.variables(inertial_coords, mesh, inv_jacobian,
                                  typename system::background_fields{});
      FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
          solution, mesh, coord_map, 1.e-7,
          tuples::apply<typename system::fluxes_computer::argument_tags>(
              get_items, background_fields),
          tuples::apply<typename system::sources_computer::argument_tags>(
              get_items, background_fields));
    }
    {
      INFO("Hamiltonian and lapse equations");
      using system =
          Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianAndLapse,
                                 Xcts::Geometry::Curved>;
      const auto background_fields =
          analytic_data.variables(inertial_coords, mesh, inv_jacobian,
                                  typename system::background_fields{});
      FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
          solution, mesh, coord_map, 1.e-7,
          tuples::apply<typename system::fluxes_computer::argument_tags>(
              get_items, background_fields),
          tuples::apply<typename system::sources_computer::argument_tags>(
              get_items, background_fields));
    }
    {
      INFO("Full XCTS equations");
      using system =
          Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
                                 Xcts::Geometry::Curved>;
      const auto background_fields =
          analytic_data.variables(inertial_coords, mesh, inv_jacobian,
                                  typename system::background_fields{});
      FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
          solution, mesh, coord_map, 1.e-7,
          tuples::apply<typename system::fluxes_computer::argument_tags>(
              get_items, background_fields),
          tuples::apply<typename system::sources_computer::argument_tags>(
              get_items, background_fields));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Kerr",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution(1., {{0., 0., 0.}}, {{0., 0., 0.}},
                "Kerr:\n"
                "  Mass: 1.\n"
                "  Spin: [0., 0., 0.]\n"
                "  Center: [0., 0., 0.]");
}

}  // namespace Xcts::Solutions
