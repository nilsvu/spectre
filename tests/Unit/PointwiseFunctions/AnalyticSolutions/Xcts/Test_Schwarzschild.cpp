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
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using field_tags = tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
                              Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                              gr::Tags::Shift<3, Frame::Inertial, DataVector>>;
using auxiliary_field_tags =
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>,
               ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>,
               Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataVector>>;
using background_tags =
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>,
               ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>>;
using matter_source_tags =
    tmpl::list<gr::Tags::EnergyDensity<DataVector>,
               gr::Tags::StressTrace<DataVector>,
               gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>;
using fixed_source_tags = db::wrap_tags_in<Tags::FixedSource, field_tags>;

template <Xcts::Solutions::SchwarzschildCoordinates Coords>
struct SchwarzschildProxy : Xcts::Solutions::Schwarzschild<Coords> {
  using Xcts::Solutions::Schwarzschild<Coords>::Schwarzschild;
  tuples::tagged_tuple_from_typelist<
      tmpl::append<field_tags, auxiliary_field_tags>>
  field_variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x) const
      noexcept {
    return Xcts::Solutions::Schwarzschild<Coords>::variables(
        x, tmpl::append<field_tags, auxiliary_field_tags>{});
  }
  tuples::tagged_tuple_from_typelist<background_tags> background_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<Coords>::variables(x,
                                                             background_tags{});
  }
  tuples::tagged_tuple_from_typelist<matter_source_tags>
  matter_source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<Coords>::variables(
        x, matter_source_tags{});
  }
  tuples::tagged_tuple_from_typelist<fixed_source_tags> fixed_source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<Coords>::variables(
        x, fixed_source_tags{});
  }
};

template <Xcts::Solutions::SchwarzschildCoordinates Coords>
void test_solution(const double radius_at_horizon,
                   const std::string& py_functions_suffix) {
  CAPTURE(Coords);
  const SchwarzschildProxy<Coords> solution{};
  CHECK(solution.radius_at_horizon() == approx(radius_at_horizon));
  const double inner_radius = 0.5 * radius_at_horizon;
  const double outer_radius = 2. * radius_at_horizon;
  pypp::check_with_random_values<
      1, tmpl::append<field_tags, auxiliary_field_tags>>(
      &SchwarzschildProxy<Coords>::field_variables, solution, "Schwarzschild",
      {"conformal_factor_" + py_functions_suffix,
       "lapse_times_conformal_factor_" + py_functions_suffix,
       "shift_" + py_functions_suffix,
       "conformal_factor_gradient_" + py_functions_suffix,
       "lapse_times_conformal_factor_gradient_" + py_functions_suffix,
       "shift_strain_" + py_functions_suffix},
      {{{inner_radius, outer_radius}}}, std::tuple<>{}, DataVector(5));
  pypp::check_with_random_values<1, background_tags>(
      &SchwarzschildProxy<Coords>::background_variables, solution,
      "Schwarzschild",
      {"extrinsic_curvature_trace_" + py_functions_suffix,
       "extrinsic_curvature_trace_gradient_" + py_functions_suffix},
      {{{inner_radius, outer_radius}}}, std::tuple<>{}, DataVector(5));
  pypp::check_with_random_values<1, matter_source_tags>(
      &SchwarzschildProxy<Coords>::matter_source_variables, solution,
      "Schwarzschild", {"energy_density", "stress_trace", "momentum_density"},
      {{{inner_radius, outer_radius}}}, std::tuple<>{}, DataVector(5));
  pypp::check_with_random_values<1, fixed_source_tags>(
      &SchwarzschildProxy<Coords>::fixed_source_variables, solution,
      "Schwarzschild",
      {"conformal_factor_fixed_source",
       "lapse_times_conformal_factor_fixed_source", "shift_fixed_source"},
      {{{inner_radius, outer_radius}}}, std::tuple<>{}, DataVector(5));

  Xcts::Solutions::Schwarzschild<Coords> created_solution =
      TestHelpers::test_creation<Xcts::Solutions::Schwarzschild<Coords>>("");
  CHECK(created_solution == solution);
  test_serialization(solution);

  {
    INFO("Verify the solution solves the XCTS system");
    using system =
        Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
                               Xcts::Geometry::Flat>;
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
    const auto matter_sources =
        solution.variables(inertial_coords, matter_source_tags{});
    const auto background_fields =
        solution.variables(inertial_coords, background_tags{});
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
        solution, typename system::fluxes{}, mesh, coord_map, 5.e-4,
        std::tuple<>{},
        std::make_tuple(
            get<gr::Tags::EnergyDensity<DataVector>>(matter_sources),
            get<gr::Tags::StressTrace<DataVector>>(matter_sources),
            get<gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>(
                matter_sources),
            get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
                background_fields),
            get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                              tmpl::size_t<3>, Frame::Inertial>>(
                background_fields)));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Schwarzschild",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution<Xcts::Solutions::SchwarzschildCoordinates::Isotropic>(
      0.5, "isotropic");
  test_solution<Xcts::Solutions::SchwarzschildCoordinates::PainleveGullstrand>(
      2., "painleve_gullstrand");
  test_solution<Xcts::Solutions::SchwarzschildCoordinates::KerrSchildIsotropic>(
      1.2727410334221052, "kerr_schild_isotropic");
}
