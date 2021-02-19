// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
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
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace {

using field_tags =
    tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
               Xcts::Tags::LapseTimesConformalFactor<DataVector>,
               Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>;
using auxiliary_field_tags =
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>,
               ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>,
               Xcts::Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>;
using background_tags =
    tmpl::list<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
               gr::Tags::TraceExtrinsicCurvature<DataVector>,
               Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>;
using matter_source_tags =
    tmpl::list<gr::Tags::EnergyDensity<DataVector>,
               gr::Tags::StressTrace<DataVector>,
               gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>;
using fixed_source_tags = db::wrap_tags_in<::Tags::FixedSource, field_tags>;

struct SchwarzschildProxy : Xcts::Solutions::Schwarzschild<> {
  using Xcts::Solutions::Schwarzschild<>::Schwarzschild;
  tuples::tagged_tuple_from_typelist<
      tmpl::append<field_tags, auxiliary_field_tags>>
  field_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(
        x, tmpl::append<field_tags, auxiliary_field_tags>{});
  }
  tuples::tagged_tuple_from_typelist<background_tags> background_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, background_tags{});
  }
  tuples::tagged_tuple_from_typelist<matter_source_tags>
  matter_source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, matter_source_tags{});
  }
  tuples::tagged_tuple_from_typelist<fixed_source_tags> fixed_source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, fixed_source_tags{});
  }
};

void test_solution(const double mass,
                   const Xcts::Solutions::SchwarzschildCoordinates coords,
                   const double expected_radius_at_horizon,
                   const std::string& py_functions_suffix,
                   const std::string& options_string) {
  CAPTURE(mass);
  CAPTURE(coords);
  const auto created =
      TestHelpers::test_factory_creation<Xcts::Solutions::AnalyticSolution<
          tmpl::list<Xcts::Solutions::Registrars::Schwarzschild>>>(
          options_string);
  {
    INFO("Semantics");
    REQUIRE(dynamic_cast<const Schwarzschild<>*>(created.get()) != nullptr);
    const auto& solution = dynamic_cast<const Schwarzschild<>&>(*created);
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }

  const SchwarzschildProxy solution{mass, coords};
  CHECK(solution.mass() == mass);
  REQUIRE(solution.radius_at_horizon() == approx(expected_radius_at_horizon));
  const double inner_radius = 0.5 * expected_radius_at_horizon;
  const double outer_radius = 2. * expected_radius_at_horizon;
  pypp::check_with_random_values<1>(
      &SchwarzschildProxy::field_variables, solution, "Schwarzschild",
      {"conformal_factor_" + py_functions_suffix,
       "lapse_times_conformal_factor_" + py_functions_suffix,
       "shift_" + py_functions_suffix,
       "conformal_factor_gradient_" + py_functions_suffix,
       "lapse_times_conformal_factor_gradient_" + py_functions_suffix,
       "shift_strain_" + py_functions_suffix},
      {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
  pypp::check_with_random_values<1>(
      &SchwarzschildProxy::background_variables, solution, "Schwarzschild",
      {"conformal_spatial_metric_" + py_functions_suffix,
       "extrinsic_curvature_trace_" + py_functions_suffix,
       "shift_background"},
      {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
  pypp::check_with_random_values<1>(
      &SchwarzschildProxy::matter_source_variables, solution, "Schwarzschild",
      {"energy_density", "stress_trace", "momentum_density"},
      {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
  pypp::check_with_random_values<1>(
      &SchwarzschildProxy::fixed_source_variables, solution, "Schwarzschild",
      {"conformal_factor_fixed_source",
       "lapse_times_conformal_factor_fixed_source", "shift_fixed_source"},
      {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));

  {
    INFO("Verify the solution solves the XCTS system");
    const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
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
    const auto get_items = [](const auto... args) {
      return std::forward_as_tuple(args...);
    };
    {
      INFO("Hamiltonian constraint only");
      using system = Xcts::FirstOrderSystem<Xcts::Equations::Hamiltonian,
                                            Xcts::Geometry::FlatCartesian>;
      const auto& data = dynamic_cast<const Xcts::AnalyticData::AnalyticData<
          tmpl::list<Registrars::Schwarzschild>>&>(solution);
      const auto background_fields =
          data.variables(inertial_coords, mesh, inv_jacobian, typename system::background_fields{});
      FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
          solution, mesh, coord_map, 1.e-3,
          apply<typename system::fluxes_computer::argument_tags>(
              get_items, background_fields),
          apply<typename system::sources_computer::argument_tags>(
              get_items, background_fields));
    }
  //   {
  //     INFO("Hamiltonian and lapse equations");
  //     using system =
  //         Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianAndLapse,
  //                                Xcts::Geometry::Euclidean>;
  //     FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
  //         solution, typename system::fluxes{}, mesh, coord_map, 1.e-3,
  //         std::tuple<>{},
  //         std::make_tuple(
  //             get<gr::Tags::EnergyDensity<DataVector>>(matter_sources),
  //             get<gr::Tags::StressTrace<DataVector>>(matter_sources),
  //             get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
  //                 background_fields),
  //             get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
  //                 background_fields),
  //             get<Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<
  //                 DataVector>>(background_fields),
  //             get<Xcts::Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>>(
  //                 background_fields)));
  //   }
  //   {
  //     INFO("Full XCTS equations");
  //     using system =
  //         Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
  //                                Xcts::Geometry::Euclidean>;
  //     FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
  //         solution, typename system::fluxes{}, mesh, coord_map, 1.e-3,
  //         std::tuple<>{},
  //         std::make_tuple(
  //             get<gr::Tags::EnergyDensity<DataVector>>(matter_sources),
  //             get<gr::Tags::StressTrace<DataVector>>(matter_sources),
  //             get<gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>(
  //                 matter_sources),
  //             get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
  //                 background_fields),
  //             get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
  //                 background_fields),
  //             get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
  //                               tmpl::size_t<3>, Frame::Inertial>>(
  //                 background_fields),
  //             get<Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>(
  //                 background_fields),
  //             get<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
  //                 DataVector, 3, Frame::Inertial>>(background_fields),
  //             get<::Tags::div<
  //                 Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
  //                     DataVector, 3, Frame::Inertial>>>(background_fields)));
  //   }
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Schwarzschild",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution(1., SchwarzschildCoordinates::Isotropic, 0.5, "isotropic",
                "Schwarzschild:\n"
                "  Mass: 1.\n"
                "  Coordinates: Isotropic");
  test_solution(0.8, SchwarzschildCoordinates::Isotropic, 0.4, "isotropic",
                "Schwarzschild:\n"
                "  Mass: 0.8\n"
                "  Coordinates: Isotropic");
}

}  // namespace Xcts::Solutions
