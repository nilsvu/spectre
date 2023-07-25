// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BlastWave.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

struct BlastWaveProxy : grmhd::AnalyticData::BlastWave {
  using grmhd::AnalyticData::BlastWave::BlastWave;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::ElectronFraction<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const {
    return variables(x, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const {
    return variables(x, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() {
  register_classes_with_charm<grmhd::AnalyticData::BlastWave>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::BlastWave>(
          "BlastWave:\n"
          "  InnerRadius: 0.8\n"
          "  OuterRadius: 1.0\n"
          "  InnerDensity: 1.0e-2\n"
          "  OuterDensity: 1.0e-4\n"
          "  InnerPressure: 1.0\n"
          "  OuterPressure: 5.0e-4\n"
          "  MagneticField: [0.1, 0.0, 0.0]\n"
          "  AdiabaticIndex: 1.3333333333333333333\n"
          "  Geometry: Cylindrical\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& cylindrical_blast_wave =
      dynamic_cast<const grmhd::AnalyticData::BlastWave&>(
          *deserialized_option_solution);

  CHECK(cylindrical_blast_wave ==
        grmhd::AnalyticData::BlastWave(
            0.8, 1.0, 1.0e-2, 1.0e-4, 1.0, 5.0e-4,
            std::array<double, 3>{{0.1, 0.0, 0.0}}, 1.3333333333333333333,
            grmhd::AnalyticData::BlastWave::Geometry::Cylindrical));
}

void test_move() {
  grmhd::AnalyticData::BlastWave cylindrical_blast_wave(
      0.8, 1.0, 1.0e-2, 1.0e-4, 1.0, 5.0e-4,
      std::array<double, 3>{{0.1, 0.0, 0.0}}, 1.3333333333333333333,
      grmhd::AnalyticData::BlastWave::Geometry::Cylindrical);
  grmhd::AnalyticData::BlastWave cylindrical_blast_wave_copy(
      0.8, 1.0, 1.0e-2, 1.0e-4, 1.0, 5.0e-4,
      std::array<double, 3>{{0.1, 0.0, 0.0}}, 1.3333333333333333333,
      grmhd::AnalyticData::BlastWave::Geometry::Cylindrical);
  test_move_semantics(std::move(cylindrical_blast_wave),
                      cylindrical_blast_wave_copy);  //  NOLINT

  grmhd::AnalyticData::BlastWave spherical_blast_wave(
      0.8, 1.0, 1.0e-2, 1.0e-4, 1.0, 5.0e-4,
      std::array<double, 3>{{0.1, 0.0, 0.0}}, 1.3333333333333333333,
      grmhd::AnalyticData::BlastWave::Geometry::Spherical);
  grmhd::AnalyticData::BlastWave spherical_blast_wave_copy(
      0.8, 1.0, 1.0e-2, 1.0e-4, 1.0, 5.0e-4,
      std::array<double, 3>{{0.1, 0.0, 0.0}}, 1.3333333333333333333,
      grmhd::AnalyticData::BlastWave::Geometry::Spherical);
  test_move_semantics(std::move(spherical_blast_wave),
                      spherical_blast_wave_copy);  //  NOLINT
}

void test_serialize() {
  grmhd::AnalyticData::BlastWave cylindrical_blast_wave(
      0.8, 1.0, 1.0e-2, 1.0e-4, 1.0, 5.0e-4,
      std::array<double, 3>{{0.1, 0.0, 0.0}}, 1.3333333333333333333,
      grmhd::AnalyticData::BlastWave::Geometry::Cylindrical);
  test_serialization(cylindrical_blast_wave);
}

template <typename DataType>
void test_variables(const DataType& used_for_size,
                    const grmhd::AnalyticData::BlastWave::Geometry geometry) {
  const double inner_radius = 0.8;
  const double outer_radius = 1.0;
  const double inner_density = 1.0e-2;
  const double outer_density = 1.0e-4;
  const double inner_pressure = 1.0;
  const double outer_pressure = 5.0e-4;
  const std::array<double, 3> magnetic_field{{0.1, 0.0, 0.0}};
  const double adiabatic_index = 1.3333333333333333333;

  // last argument is 1.0 if Cylindrical and 0.0 if Spherical geometry
  const auto member_variables = std::make_tuple(
      inner_radius, outer_radius, inner_density, outer_density, inner_pressure,
      outer_pressure, magnetic_field, adiabatic_index,
      geometry == grmhd::AnalyticData::BlastWave::Geometry::Cylindrical ? 1.0
                                                                        : 0.0);

  BlastWaveProxy blast_wave(inner_radius, outer_radius, inner_density,
                            outer_density, inner_pressure, outer_pressure,
                            magnetic_field, adiabatic_index, geometry);

  // Note: I select random numbers in the range {{-1.1, 1.1}} so that
  // sometimes the random points are in the transition region and sometimes
  // in the fixed region.
  pypp::check_with_random_values<1>(
      &BlastWaveProxy::hydro_variables<DataType>, blast_wave, "BlastWave",
      {"rest_mass_density", "electron_fraction", "spatial_velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy"},
      {{{-1.1, 1.1}}}, member_variables, used_for_size);

  pypp::check_with_random_values<1>(
      &BlastWaveProxy::grmhd_variables<DataType>, blast_wave, "BlastWave",
      {"rest_mass_density", "electron_fraction", "spatial_velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy", "magnetic_field", "divergence_cleaning_field"},
      {{{-1.1, 1.1}}}, member_variables, used_for_size);
}

// Check points on and near boundaries. Check that density = inner_density
// at r = inner radius and at r = inner_radius - epsilon, and check that
// density is approximately inner_density at r = inner_radius + epsilon.
// Then do the analogous check for points on and near outer_radius.
void test_density_on_and_near_boundaries(
    const grmhd::AnalyticData::BlastWave::Geometry geometry) {
  const double inner_radius = 0.8;
  const double outer_radius = 1.0;
  const double inner_density = 1.0e-2;
  const double outer_density = 1.0e-4;
  const double inner_pressure = 1.0;
  const double outer_pressure = 5.0e-4;
  const std::array<double, 3> magnetic_field{{0.1, 0.0, 0.0}};
  const double adiabatic_index = 1.3333333333333333333;

  BlastWaveProxy blast_wave(inner_radius, outer_radius, inner_density,
                            outer_density, inner_pressure, outer_pressure,
                            magnetic_field, adiabatic_index, geometry);

  const double epsilon = 1.e-10;
  Approx approx = Approx::custom().epsilon(epsilon * 100.0);
  auto x =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(inner_radius, 0.0);

  get<0>(x) = inner_radius;
  CHECK(inner_density == get(get<hydro::Tags::RestMassDensity<double>>(
                             blast_wave.grmhd_variables(x))));
  get<0>(x) = inner_radius - epsilon;
  CHECK(inner_density == get(get<hydro::Tags::RestMassDensity<double>>(
                             blast_wave.grmhd_variables(x))));
  get<0>(x) = inner_radius + epsilon;
  CHECK_ITERABLE_CUSTOM_APPROX(inner_density,
                               get(get<hydro::Tags::RestMassDensity<double>>(
                                   blast_wave.grmhd_variables(x))),
                               approx);

  get<0>(x) = outer_radius;
  CHECK(outer_density == get(get<hydro::Tags::RestMassDensity<double>>(
                             blast_wave.grmhd_variables(x))));
  get<0>(x) = outer_radius + epsilon;
  CHECK(outer_density == get(get<hydro::Tags::RestMassDensity<double>>(
                             blast_wave.grmhd_variables(x))));
  get<0>(x) = outer_radius - epsilon;
  CHECK_ITERABLE_CUSTOM_APPROX(outer_density,
                               get(get<hydro::Tags::RestMassDensity<double>>(
                                   blast_wave.grmhd_variables(x))),
                               approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.BlastWave",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();

  for (const auto geometry :
       {grmhd::AnalyticData::BlastWave::Geometry::Cylindrical,
        grmhd::AnalyticData::BlastWave::Geometry::Spherical}) {
    test_variables(std::numeric_limits<double>::signaling_NaN(), geometry);
    test_variables(DataVector(5), geometry);
    test_density_on_and_near_boundaries(geometry);
  }

  CHECK_THROWS_WITH(
      (grmhd::AnalyticData::BlastWave(
          1.2, 1.0, 1.0e-2, 1.0e-4, 1.0, 5.0e-4,
          std::array<double, 3>{{0.1, 0.0, 0.0}}, 1.3333333333333333333,
          grmhd::AnalyticData::BlastWave::Geometry::Cylindrical)),
      Catch::Matchers::Contains("BlastWave expects InnerRadius < OuterRadius"));
}
