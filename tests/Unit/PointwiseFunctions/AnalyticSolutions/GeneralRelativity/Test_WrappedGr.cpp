// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
template <typename SolutionType>
void compare_different_wrapped_solutions(const double mass,
                                         const std::array<double, 3>& spin,
                                         const std::array<double, 3>& center,
                                         const double mass2,
                                         const std::array<double, 3>& spin2,
                                         const std::array<double, 3>& center2) {
  const SolutionType& solution{mass, spin, center};
  const SolutionType& solution2{mass2, spin2, center2};
  CHECK_FALSE(solution == solution2);
  CHECK(solution != solution2);
}

template <typename SolutionType>
void test_copy_and_move(const SolutionType& solution) {
  test_copy_semantics(solution);
  auto solution_copy = solution;
  auto solution_copy2 = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution_copy2), solution_copy);  // NOLINT
}

template <typename SolutionType, typename... Args>
void test_generalized_harmonic_solution(const Args&... args) {
  const SolutionType& solution{args...};
  const gh::Solutions::WrappedGr<SolutionType>& wrapped_solution{args...};

  const DataVector data_vector{3.0, 4.0};
  const tnsr::I<DataVector, SolutionType::volume_dim, Frame::Inertial> x{
      data_vector};
  // Don't set time to signaling NaN, since not all solutions tested here
  // are static
  const double t = 44.44;

  // Check that the wrapped solution returns the same variables as
  // the solution
  const auto vars = solution.variables(
      x, t, typename SolutionType::template tags<DataVector>{});
  const auto wrapped_vars = wrapped_solution.variables(
      x, t, typename SolutionType::template tags<DataVector>{});

  tmpl::for_each<typename SolutionType::template tags<DataVector>>(
      [&vars, &wrapped_vars](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        CHECK(get<tag>(vars) == get<tag>(wrapped_vars));
      });

  // Check that the wrapped solution returns the correct psi, pi, phi
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto& d_lapse =
      get<Tags::deriv<gr::Tags::Lapse<DataVector>,
                      tmpl::size_t<SolutionType::volume_dim>, Frame::Inertial>>(
          vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, SolutionType::volume_dim>>(vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataVector, SolutionType::volume_dim>>>(
          vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<DataVector, SolutionType::volume_dim>,
                      tmpl::size_t<SolutionType::volume_dim>, Frame::Inertial>>(
          vars);
  const auto& g =
      get<gr::Tags::SpatialMetric<DataVector, SolutionType::volume_dim>>(vars);
  const auto& dt_g = get<
      Tags::dt<gr::Tags::SpatialMetric<DataVector, SolutionType::volume_dim>>>(
      vars);
  const auto& d_g = get<
      Tags::deriv<gr::Tags::SpatialMetric<DataVector, SolutionType::volume_dim>,
                  tmpl::size_t<SolutionType::volume_dim>, Frame::Inertial>>(
      vars);
  const auto psi = gr::spacetime_metric(lapse, shift, g);
  const auto phi = gh::phi(lapse, d_lapse, shift, d_shift, g, d_g);
  const auto pi = gh::pi(lapse, dt_lapse, shift, dt_shift, g, dt_g, phi);

  const auto wrapped_gh_vars = wrapped_solution.variables(
      x, t,
      tmpl::list<
          gr::Tags::SpacetimeMetric<DataVector, SolutionType::volume_dim>,
          gh::Tags::Pi<DataVector, SolutionType::volume_dim>,
          gh::Tags::Phi<DataVector, SolutionType::volume_dim>>{});
  CHECK(psi ==
        get<gr::Tags::SpacetimeMetric<DataVector, SolutionType::volume_dim>>(
            wrapped_gh_vars));
  CHECK(pi == get<gh::Tags::Pi<DataVector, SolutionType::volume_dim>>(
                  wrapped_gh_vars));
  CHECK(phi == get<gh::Tags::Phi<DataVector, SolutionType::volume_dim>>(
                   wrapped_gh_vars));

  // Weak test of operators == and !=
  CHECK(wrapped_solution == wrapped_solution);
  CHECK_FALSE(wrapped_solution != wrapped_solution);

  test_serialization(wrapped_solution);
  test_copy_and_move(wrapped_solution);
}

template <typename SolutionType>
void test_construct_from_options() {
  const auto created =
      TestHelpers::test_creation<gh::Solutions::WrappedGr<SolutionType>>(
          "Mass: 0.5\n"
          "Spin: [0.1,0.2,0.3]\n"
          "Center: [1.0,3.0,2.0]\n"
          "Velocity: [0,0,0]");
  const double mass = 0.5;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{1.0, 3.0, 2.0}};
  CHECK(created == gh::Solutions::WrappedGr<SolutionType>(mass, spin, center));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.WrappedGr",
                  "[PointwiseFunctions][Unit]") {
  const double amplitude = 0.24;
  const double wavelength = 4.4;
  test_generalized_harmonic_solution<gr::Solutions::GaugeWave<1>>(amplitude,
                                                                  wavelength);
  test_generalized_harmonic_solution<gr::Solutions::GaugeWave<2>>(amplitude,
                                                                  wavelength);
  test_generalized_harmonic_solution<gr::Solutions::GaugeWave<3>>(amplitude,
                                                                  wavelength);
  test_generalized_harmonic_solution<gr::Solutions::Minkowski<1>>();
  test_generalized_harmonic_solution<gr::Solutions::Minkowski<2>>();
  test_generalized_harmonic_solution<gr::Solutions::Minkowski<3>>();

  const double mass = 0.5;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{1.0, 3.0, 2.0}};
  test_generalized_harmonic_solution<gr::Solutions::KerrSchild>(mass, spin,
                                                                center);

  test_generalized_harmonic_solution<gr::Solutions::SphericalKerrSchild>(
      mass, spin, center);

  const double mass2 = 0.4;
  const std::array<double, 3> spin2{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center2{{4.0, 1.0, 3.0}};
  test_generalized_harmonic_solution<gr::Solutions::KerrSchild>(mass2, spin2,
                                                                center2);

  test_generalized_harmonic_solution<gr::Solutions::SphericalKerrSchild>(
      mass2, spin2, center2);

  compare_different_wrapped_solutions<gr::Solutions::KerrSchild>(
      mass, spin, center, mass2, spin2, center2);

  compare_different_wrapped_solutions<gr::Solutions::SphericalKerrSchild>(
      mass, spin, center, mass2, spin2, center2);

  test_construct_from_options<gr::Solutions::KerrSchild>();
}
