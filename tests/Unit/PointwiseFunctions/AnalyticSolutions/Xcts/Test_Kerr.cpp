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
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using background_tags = tmpl::list<
    gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
    gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
    Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
    Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>,
    Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataVector, 3, Frame::Inertial>>;

template <Xcts::Solutions::KerrCoordinates Coords>
void test_solution() {
  CAPTURE(Coords);

  const Xcts::Solutions::Kerr<Coords> solution{
      1., {{0.1, 0.2, 0.3}}, {{0., 0., 0.}}};
  const auto created_solution =
      TestHelpers::test_creation<Xcts::Solutions::Kerr<Coords>>(
          "Mass: 1.\n"
          "Spin: [0.1, 0.2, 0.3]\n"
          "Center: [0., 0., 0.]");
  CHECK(created_solution == solution);
  test_serialization(solution);

  const double inner_radius = 2.2;
  const double outer_radius = 5.;

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
    const auto background_fields =
        solution.variables(inertial_coords, background_tags{});
    const auto& conformal_metric =
        get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& inv_conformal_metric =
        get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& conformal_metric_deriv = get<::Tags::deriv<
        Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(background_fields);
    const auto conformal_christoffel_first_kind =
        gr::christoffel_first_kind(conformal_metric_deriv);
    const auto conformal_christoffel_second_kind = raise_or_lower_first_index(
        conformal_christoffel_first_kind, inv_conformal_metric);
    auto conformal_christoffel_contracted =
        make_with_value<tnsr::i<DataVector, 3>>(inertial_coords, 0.);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        conformal_christoffel_contracted.get(i) +=
            conformal_christoffel_second_kind.get(j, i, j);
      }
    }
    Variables<tmpl::list<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
        gr::Tags::TraceExtrinsicCurvature<DataVector>>>
        background_fields_to_derive{mesh.number_of_grid_points()};
    get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>>(
        background_fields_to_derive) = conformal_christoffel_second_kind;
    get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
        background_fields_to_derive) =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields);
    const auto deriv_background_fields = partial_derivatives<tmpl::list<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
        gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
        background_fields_to_derive, mesh, inv_jacobian);
    const auto& deriv_conformal_christoffel_second_kind = get<::Tags::deriv<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
        tmpl::size_t<3>, Frame::Inertial>>(deriv_background_fields);
    const auto conformal_ricci_tensor =
        gr::ricci_tensor(conformal_christoffel_second_kind,
                         deriv_conformal_christoffel_second_kind);
    const auto conformal_ricci_scalar =
        trace(conformal_ricci_tensor, inv_conformal_metric);
    const auto& shift_background =
        get<Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>(
            background_fields);
    auto longitudinal_shift_background_minus_dt_conformal_metric =
        get<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataVector, 3, Frame::Inertial>>(background_fields);
    {
      const auto vars = solution.variables(
          inertial_coords,
          tmpl::list<
              Xcts::Tags::ConformalFactor<DataVector>,
              Xcts::Tags::LapseTimesConformalFactor<DataVector>,
              Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
              Xcts::Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>{});
      tnsr::II<DataVector, 3> longitudinal_shift_excess{
          mesh.number_of_grid_points()};
      Xcts::longitudinal_shift(
          make_not_null(&longitudinal_shift_excess), inv_conformal_metric,
          get<Xcts::Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>(vars));
      auto longitudinal_shift_minus_dt_conformal_metric =
          longitudinal_shift_background_minus_dt_conformal_metric;
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j <= i; ++j) {
          longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
              longitudinal_shift_excess.get(i, j);
        }
      }
      Scalar<DataVector> longitudinal_shift_minus_dt_conformal_metric_square{
          mesh.number_of_grid_points()};
      Xcts::fully_contract(
          make_not_null(&longitudinal_shift_minus_dt_conformal_metric_square),
          longitudinal_shift_minus_dt_conformal_metric,
          longitudinal_shift_minus_dt_conformal_metric, conformal_metric);
      {
        INFO("Hamiltonian equation only");
        using system = Xcts::FirstOrderSystem<Xcts::Equations::Hamiltonian,
                                              Xcts::Geometry::NonEuclidean>;
        auto lapse =
            get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(vars);
        get(lapse) /= get(get<Xcts::Tags::ConformalFactor<DataVector>>(vars));
        auto longitudinal_shift_minus_dt_conformal_metric_over_lapse_square =
            longitudinal_shift_minus_dt_conformal_metric_square;
        get(longitudinal_shift_minus_dt_conformal_metric_over_lapse_square) /=
            square(get(lapse));
        FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
            solution, typename system::fluxes{}, mesh, coord_map, 1.e-7,
            std::make_tuple(inv_conformal_metric),
            std::make_tuple(
                get<gr::Tags::EnergyDensity<DataVector>>(background_fields),
                get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
                    background_fields),
                longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
                conformal_christoffel_contracted, conformal_ricci_scalar));
      }
      {
        INFO("Hamiltonian and lapse equations only");
        using system =
            Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianAndLapse,
                                   Xcts::Geometry::NonEuclidean>;
        auto shift = shift_background;
        for (size_t i = 0; i < 3; ++i) {
          shift.get(i) +=
              get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(vars)
                  .get(i);
        }
        FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
            solution, typename system::fluxes{}, mesh, coord_map, 1.e-7,
            std::make_tuple(inv_conformal_metric),
            std::make_tuple(
                get<gr::Tags::EnergyDensity<DataVector>>(background_fields),
                get<gr::Tags::StressTrace<DataVector>>(background_fields),
                get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
                    background_fields),
                get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
                    background_fields),
                longitudinal_shift_minus_dt_conformal_metric_square,
                dot_product(shift,
                            get<::Tags::deriv<
                                gr::Tags::TraceExtrinsicCurvature<DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(
                                deriv_background_fields)),
                conformal_christoffel_contracted, conformal_ricci_scalar));
      }
    }
    {
      INFO("Full XCTS equations");
      Variables<tmpl::list<
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataVector, 3, Frame::Inertial>>>
          background_fields_for_divergence{mesh.number_of_grid_points()};
      get<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>(background_fields_for_divergence) =
          longitudinal_shift_background_minus_dt_conformal_metric;
      auto div_background_fields =
          divergence(background_fields_for_divergence, mesh, inv_jacobian);
      using system =
          Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
                                 Xcts::Geometry::NonEuclidean>;
      FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
          solution, typename system::fluxes{}, mesh, coord_map, 1.e-7,
          std::make_tuple(conformal_metric, inv_conformal_metric),
          std::make_tuple(
              get<gr::Tags::EnergyDensity<DataVector>>(background_fields),
              get<gr::Tags::StressTrace<DataVector>>(background_fields),
              get<gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>(
                  background_fields),
              get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
                  background_fields),
              get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
                  background_fields),
              get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(
                  deriv_background_fields),
              shift_background,
              longitudinal_shift_background_minus_dt_conformal_metric,
              get<::Tags::div<
                  Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataVector, 3, Frame::Inertial>>>(div_background_fields),
              conformal_metric, inv_conformal_metric,
              conformal_christoffel_first_kind,
              conformal_christoffel_second_kind,
              conformal_christoffel_contracted, conformal_ricci_scalar));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Kerr",
                  "[PointwiseFunctions][Unit]") {
  test_solution<Xcts::Solutions::KerrCoordinates::KerrSchild>();
}
