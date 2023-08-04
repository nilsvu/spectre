// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include <pup.h>

namespace {

template <size_t Dim>
struct IsentropicVortexProxy
    : NewtonianEuler::Solutions::IsentropicVortex<Dim> {
  using NewtonianEuler::Solutions::IsentropicVortex<Dim>::IsentropicVortex;

  template <typename DataType>
  using variables_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                 NewtonianEuler::Tags::Velocity<DataType, Dim>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                 NewtonianEuler::Tags::Pressure<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                      double t) const {
    return this->variables(x, t, variables_tags<DataType>{});
  }
};

template <size_t Dim, typename DataType>
void test_solution(const DataType& used_for_size,
                   const std::array<double, Dim>& center,
                   const std::string& center_option,
                   const std::array<double, Dim>& mean_velocity,
                   const std::string& mean_velocity_option,
                   const double perturbation_amplitude = 0.0,
                   const std::string& perturbation_amplitude_option = "") {
  IsentropicVortexProxy<Dim> vortex(1.43, center, mean_velocity, 3.76,
                                    perturbation_amplitude);
  pypp::check_with_random_values<1>(
      &IsentropicVortexProxy<Dim>::template primitive_variables<DataType>,
      vortex, "IsentropicVortex",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{-1., 1.}}},
      std::make_tuple(1.43, center, mean_velocity, 3.76,
                      perturbation_amplitude),
      used_for_size);

  std::string input = "  AdiabaticIndex: 1.43\n  Center: " + center_option +
                      "\n  MeanVelocity: " + mean_velocity_option +
                      "\n  Strength: 3.76";

  if (Dim == 3) {
    input += "\n  PerturbAmplitude: " + perturbation_amplitude_option;
  }

  const auto vortex_from_options = TestHelpers::test_creation<
      NewtonianEuler::Solutions::IsentropicVortex<Dim>>(input);
  CHECK(vortex_from_options == vortex);

  IsentropicVortexProxy<Dim> vortex_to_move(1.43, center, mean_velocity, 3.76,
                                            perturbation_amplitude);
  test_move_semantics(std::move(vortex_to_move), vortex);  //  NOLINT
  test_copy_semantics(vortex);

  test_serialization(vortex);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.Vortex",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};

  const std::array<double, 2> center_2d = {{0.12, -0.04}};
  const std::array<double, 2> mean_velocity_2d = {{0.54, -0.02}};
  test_solution<2>(std::numeric_limits<double>::signaling_NaN(), center_2d,
                   "[0.12, -0.04]", mean_velocity_2d, "[0.54, -0.02]");
  test_solution<2>(DataVector(5), center_2d, "[0.12, -0.04]", mean_velocity_2d,
                   "[0.54, -0.02]");

  const std::array<double, 3> center_3d = {{-0.53, -0.1, 1.4}};
  const std::array<double, 3> mean_velocity_3d = {{-0.04, 0.14, 0.3}};
  const double perturbation_amplitude = 0.5;
  test_solution<3>(std::numeric_limits<double>::signaling_NaN(), center_3d,
                   "[-0.53, -0.1, 1.4]", mean_velocity_3d, "[-0.04, 0.14, 0.3]",
                   perturbation_amplitude, "0.5");
  test_solution<3>(DataVector(5), center_3d, "[-0.53, -0.1, 1.4]",
                   mean_velocity_3d, "[-0.04, 0.14, 0.3]",
                   perturbation_amplitude, "0.5");

  static_assert(
      std::is_same_v<
          NewtonianEuler::Solutions::IsentropicVortex<3>::source_term_type,
          NewtonianEuler::Sources::VortexPerturbation>);
  static_assert(
      std::is_same_v<
          NewtonianEuler::Solutions::IsentropicVortex<2>::source_term_type,
          NewtonianEuler::Sources::NoSource>);
  NewtonianEuler::Solutions::IsentropicVortex<3> vortex(
      1.3, {{3.21, -1.4}}, {{0.12, -0.53}}, 0.05, 1.7);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const double random_z = distribution(gen);
  const DataVector random_z_dv = {{random_z, random_z - 1.0, random_z + 2.0,
                                   3.0 * random_z, exp(random_z)}};
  CHECK(vortex.perturbation_profile(random_z) == sin(random_z));
  CHECK(vortex.perturbation_profile(random_z_dv) == sin(random_z_dv));
  CHECK(vortex.deriv_of_perturbation_profile(random_z) == cos(random_z));
  CHECK(vortex.deriv_of_perturbation_profile(random_z_dv) == cos(random_z_dv));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      []() {
        NewtonianEuler::Solutions::IsentropicVortex<2> test_vortex(
            1.3, {{3.21, -1.4}}, {{0.12, -0.53}}, 1.7, 1.e-12);
      }(),
      Catch::Matchers::ContainsSubstring(
          "A nonzero perturbation amplitude only "
          "makes sense in 3 dimensions."));
  CHECK_THROWS_WITH(
      []() {
        NewtonianEuler::Solutions::IsentropicVortex<2> test_vortex(
            1.3, {{3.21, -1.4}}, {{0.12, -0.53}}, -1.7);
      }(),
      Catch::Matchers::ContainsSubstring("The strength must be non-negative."));
  CHECK_THROWS_WITH(
      []() {
        NewtonianEuler::Solutions::IsentropicVortex<3> test_vortex(
            1.65, {{-0.12, 1.542, 3.12}}, {{-0.04, -0.32, 0.003}}, -0.5, 4.2);
      }(),
      Catch::Matchers::ContainsSubstring("The strength must be non-negative."));
#endif
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        NewtonianEuler::Solutions::IsentropicVortex<2>>(
                        "AdiabaticIndex: 1.4\n"
                        "Center: [-3.9, 1.1]\n"
                        "MeanVelocity: [0.1, -0.032]\n"
                        "Strength: -0.2"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -0.2 is below the lower bound of 0"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        NewtonianEuler::Solutions::IsentropicVortex<3>>(
                        "AdiabaticIndex: 1.12\n"
                        "Center: [0.3, -0.12, 4.2]\n"
                        "MeanVelocity: [-0.03, -0.1, 0.09]\n"
                        "Strength: -0.3\n"
                        "PerturbAmplitude: 0.42"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -0.3 is below the lower bound of 0"));
}
