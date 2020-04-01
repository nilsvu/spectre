// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

using field_tags = tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>;
using auxiliary_field_tags =
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>>;
using background_tags = tmpl::list<
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
        DataVector>,
    gr::Tags::EnergyDensity<DataVector>>;
using initial_tags =
    db::wrap_tags_in<Tags::Initial,
                     tmpl::append<field_tags, auxiliary_field_tags>>;
using source_tags = db::wrap_tags_in<Tags::FixedSource, field_tags>;

struct ConstantDensityStarProxy : Xcts::Solutions::ConstantDensityStar {
  using Xcts::Solutions::ConstantDensityStar::ConstantDensityStar;
  tuples::tagged_tuple_from_typelist<
      tmpl::append<field_tags, auxiliary_field_tags>>
  field_variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x) const
      noexcept {
    return Xcts::Solutions::ConstantDensityStar::variables(
        x, tmpl::append<field_tags, auxiliary_field_tags>{});
  }
  tuples::tagged_tuple_from_typelist<background_tags> background_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::ConstantDensityStar::variables(x,
                                                           background_tags{});
  }
  tuples::tagged_tuple_from_typelist<initial_tags> initial_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::ConstantDensityStar::variables(x, initial_tags{});
  }
  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::ConstantDensityStar::variables(x, source_tags{});
  }
};

void test_solution(const double density, const double radius,
                   const std::string& options) {
  const ConstantDensityStarProxy solution(density, radius);
  const double test_radius = 2. * radius;
  pypp::check_with_random_values<1, field_tags>(
      &ConstantDensityStarProxy::field_variables, solution,
      "ConstantDensityStar", {"conformal_factor", "conformal_factor_gradient"},
      {{{-test_radius, test_radius}}}, std::make_tuple(density, radius),
      DataVector(5));
  pypp::check_with_random_values<1, background_tags>(
      &ConstantDensityStarProxy::background_variables, solution,
      "ConstantDensityStar",
      {"extrinsic_curvature_trace",
       "longitudinal_shift_minus_dt_conformal_metric_over_lapse_square",
       "energy_density"},
      {{{-test_radius, test_radius}}}, std::make_tuple(density, radius),
      DataVector(5));
  pypp::check_with_random_values<1, initial_tags>(
      &ConstantDensityStarProxy::initial_variables, solution,
      "ConstantDensityStar",
      {"initial_conformal_factor", "initial_conformal_factor_gradient"},
      {{{-test_radius, test_radius}}}, std::make_tuple(density, radius),
      DataVector(5));
  pypp::check_with_random_values<1, source_tags>(
      &ConstantDensityStarProxy::source_variables, solution,
      "ConstantDensityStar", {"conformal_factor_source"},
      {{{-test_radius, test_radius}}}, std::make_tuple(density, radius),
      DataVector(5));

  // Test that we selected the weak-field solution of the two possible branches
  const tnsr::I<DataVector, 3> far_away_coords{{{{1e16 * radius, 0., 0.},
                                                 {0., 1e16 * radius, 0.},
                                                 {0., 0., 1e16 * radius}}}};
  const auto far_away_solution = solution.field_variables(far_away_coords);
  CHECK_ITERABLE_APPROX(
      get<Xcts::Tags::ConformalFactor<DataVector>>(far_away_solution),
      make_with_value<Scalar<DataVector>>(far_away_coords, 1.));

  const auto created_solution =
      TestHelpers::test_creation<Xcts::Solutions::ConstantDensityStar>(options);
  CHECK(created_solution == solution);
  test_serialization(solution);

  {
    INFO("Verify the solution solves the XCTS system");
    using system = Xcts::FirstOrderSystem<Xcts::Equations::Hamiltonian,
                                          Xcts::Geometry::Euclidean>;
    const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{{{-1., 1., 0., test_radius},
                   {-1., 1., 0., test_radius},
                   {-1., 1., 0., test_radius}}};
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = coord_map(logical_coords);
    const auto background_fields = solution.variables(
        inertial_coords, typename system::background_fields{});
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
        solution, typename system::fluxes{}, mesh, coord_map, 5.e-2,
        std::tuple<>{},
        std::make_tuple(
            get<gr::Tags::EnergyDensity<DataVector>>(background_fields),
            get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
                background_fields),
            get<Xcts::Tags::
                    LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
                        DataVector>>(background_fields)));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.ConstantDensityStar",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution(0.01, 1.,
                "Density: 0.01\n"
                "Radius: 1.\n");
}
