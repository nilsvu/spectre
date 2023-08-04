// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <size_t Dim>
struct GaugeWaveProxy : gr::Solutions::GaugeWave<Dim> {
  using gr::Solutions::GaugeWave<Dim>::GaugeWave;

  template <typename DataType>
  using variables_tags =
      typename gr::Solutions::GaugeWave<Dim>::template tags<DataType>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>> test_variables(
      const tnsr::I<DataType, Dim>& x, const double t) const {
    return this->variables(x, t, variables_tags<DataType>{});
  }
};

template <size_t Dim, typename DataType>
tnsr::I<DataType, Dim, Frame::Inertial> spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, Dim, Frame::Inertial>>(
      used_for_size, 0.0);
  get<0>(x) = 1.32;
  if (Dim >= 2) {
    x.get(1) = 0.82;
  }
  if (Dim >= 3) {
    x.get(2) = 1.24;
  }
  return x;
}

template <size_t Dim, typename DataType>
void test_gauge_wave(const gr::Solutions::GaugeWave<Dim>& solution,
                     const DataType& used_for_size) {
  using GaugeWave = gr::Solutions::GaugeWave<Dim>;
  // The solution is as follows, with H = 1 - A \sin((2*\pi/d)*(x-t))
  // and \partial_x H = - A (2*\pi/d) \cos((2*\pi/d)*(x-t)):
  //
  // \gamma_{ij} = 0, except \gamma_{xx} = H and \gamma_{yy} = \gamma_{zz} = 1
  // \gamma^{ij} = 0, except \gamma^{xx} = H and \gamma^{yy} = \gamma^{zz} = 1
  // \partial_t\gamma_{ij} = 0, except \partial_t\gamma_{xx} = - \partial_x H
  // \partial_x\gamma_{ij} = 0, except \partial_x\gamma_{xx} = \partial_x H
  // \sqrt{\gamma} = \sqrt{H}
  // \alpha = \sqrt{H}
  // \partial_t \alpha = - \partial_x H / (2 \sqrt{H})
  // \partial_x \alpha = \partial_x H / (2 \sqrt{H})
  // \partial_y \alpha = \partial_z \alpha = 0
  // \beta^i = 0
  // \partial_t \beta^i = 0
  // \partial_i \beta^j = 0
  // K_{ij} = 0, except K_{xx} = \partial_x H / (2 \sqrt{H})

  // Parameters for GaugeWave solution
  const double amplitude = 0.24;
  const double wavelength = 4.4;

  // Verify that the solution passed in was constructed with the correct
  // parameters
  REQUIRE(solution == GaugeWave{amplitude, wavelength});

  // Using `auto` triggers a gcc bug (known broken in 6.5.0, working in 9.2.0)
  const tnsr::I<DataType, Dim, Frame::Inertial> x =
      spatial_coords<Dim>(used_for_size);
  const double t = 1.3;

  // Using `auto` triggers a gcc bug (known broken in 6.5.0, working in 9.2.0)
  const tuples::tagged_tuple_from_typelist<
      typename gr::Solutions::GaugeWave<Dim>::template tags<DataType>>
      vars = solution.variables(x, t,
                                typename GaugeWave::template tags<DataType>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& d_lapse =
      get<typename GaugeWave::template DerivLapse<DataType>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataType, Dim>>(vars);
  const auto& d_shift =
      get<typename GaugeWave::template DerivShift<DataType>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataType, Dim>>>(vars);
  const auto& gamma = get<gr::Tags::SpatialMetric<DataType, Dim>>(vars);
  const auto& dt_gamma =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, Dim>>>(vars);
  const auto& d_gamma =
      get<typename GaugeWave::template DerivSpatialMetric<DataType>>(vars);

  // Check those quantities that should be zero or one.
  const auto zero = make_with_value<DataType>(x, 0.);
  const auto one = make_with_value<DataType>(x, 1.);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(shift.get(i), zero);
    CHECK_ITERABLE_APPROX(dt_shift.get(i), zero);
    if (i > 0) {
      CHECK_ITERABLE_APPROX(d_lapse.get(i), zero);
    }
    for (size_t j = 0; j < Dim; ++j) {
      CHECK_ITERABLE_APPROX(d_shift.get(i, j), zero);
      if (i != j) {
        CHECK_ITERABLE_APPROX(gamma.get(i, j), zero);
      } else {
        if (i > 0) {
          CHECK_ITERABLE_APPROX(gamma.get(i, j), one);
        }
      }
      if (i > 0 or j > 0) {
        CHECK_ITERABLE_APPROX(dt_gamma.get(i, j), zero);
        for (size_t k = 0; k < Dim; ++k) {
          CHECK_ITERABLE_APPROX(d_gamma.get(k, i, j), zero);
        }
      } else {
        for (size_t k = 1; k < Dim; ++k) {
          CHECK_ITERABLE_APPROX(d_gamma.get(k, i, j), zero);
        }
      }
    }
  }

  // Check quantities that depend on h
  const auto expected_lapse =
      make_with_value<DataType>(get(lapse), 0.9965673824732139);
  const auto expected_dt_lapse =
      make_with_value<DataType>(get(lapse), 0.17187971493515364);
  const auto expected_d_lapse_0 =
      make_with_value<DataType>(get(lapse), -0.17187971493515364);
  CHECK_ITERABLE_APPROX(get(lapse), expected_lapse);
  CHECK_ITERABLE_APPROX(get(dt_lapse), expected_dt_lapse);
  CHECK_ITERABLE_APPROX(get<0>(d_lapse), expected_d_lapse_0);

  const auto expected_gamma_0_0 =
      make_with_value<DataType>(get(lapse), 0.993146547809513);
  const auto expected_dt_gamma_0_0 =
      make_with_value<DataType>(get(lapse), 0.34257943522633644);
  const auto expected_d_gamma_0_0_0 =
      make_with_value<DataType>(get(lapse), -0.34257943522633644);
  // CHECK_ITERABLE_APPROX breaks if you try to pass get<0, 0>(...)
  // in for argument a, becaue it doesn't like the comma. So
  // define aliases for get<0, 0>(gamma), etc. first.
  const auto& gamma_0_0 = get<0, 0>(gamma);
  const auto& dt_gamma_0_0 = get<0, 0>(dt_gamma);
  const auto& d_gamma_0_0_0 = get<0, 0, 0>(d_gamma);
  CHECK_ITERABLE_APPROX(gamma_0_0, expected_gamma_0_0);
  CHECK_ITERABLE_APPROX(dt_gamma_0_0, expected_dt_gamma_0_0);
  CHECK_ITERABLE_APPROX(d_gamma_0_0_0, expected_d_gamma_0_0_0);

  // Check quantities derivable from spatial metric, lapse, shift, and their
  // derivatives
  const auto& inv_gamma =
      get<gr::Tags::InverseSpatialMetric<DataType, Dim>>(vars);
  const auto& sqrt_det_gamma =
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(vars);
  const auto& det_and_inverse = determinant_and_inverse(gamma);
  const auto& expected_inv_gamma = det_and_inverse.second;
  const auto& expected_sqrt_det_gamma = sqrt(get(det_and_inverse.first));
  CHECK_ITERABLE_APPROX(inv_gamma, expected_inv_gamma);
  CHECK_ITERABLE_APPROX(get(sqrt_det_gamma), expected_sqrt_det_gamma);

  const auto& K = get<gr::Tags::ExtrinsicCurvature<DataType, Dim>>(vars);
  const auto& expected_K =
      gr::extrinsic_curvature(lapse, shift, d_shift, gamma, dt_gamma, d_gamma);
  CHECK_ITERABLE_APPROX(K, expected_K);

  // Check that we can retrieve tags individually from the solution
  tmpl::for_each<typename GaugeWave::template tags<DataType>>(
      [&solution, &vars, &x, &t](auto tag_v) {
        using Tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<Tag>(vars), get<Tag>(solution.variables(
                                                  x, t, tmpl::list<Tag>{})));
      });

  // Check with random values
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/"};
  pypp::check_with_random_values<1>(
      &GaugeWaveProxy<Dim>::template test_variables<DataType>,
      GaugeWaveProxy<Dim>(amplitude, wavelength), "GaugeWave",
      {"gauge_wave_lapse", "gauge_wave_dt_lapse", "gauge_wave_d_lapse",
       "gauge_wave_shift", "gauge_wave_dt_shift", "gauge_wave_d_shift",
       "gauge_wave_spatial_metric", "gauge_wave_dt_spatial_metric",
       "gauge_wave_d_spatial_metric", "gauge_wave_sqrt_det_spatial_metric",
       "gauge_wave_extrinsic_curvature", "gauge_wave_inverse_spatial_metric"},
      {{{1.0, 20.0}}}, std::make_tuple(amplitude, wavelength), used_for_size);

  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t,
      typename gr::Solutions::GaugeWave<Dim>::template tags<DataType>{});
}

void test_consistency() {
  const gr::Solutions::GaugeWave<3> solution(0.3, 2.4);
  TestHelpers::VerifyGrSolution::verify_consistency(
      solution, 1.234, tnsr::I<double, 3>{{{1.2, 2.3, 3.4}}}, 0.01, 1.0e-8);
}

void test_serialize() {
  const gr::Solutions::GaugeWave<3> solution(0.24, 4.4);
  test_serialization(solution);
  test_gauge_wave(serialize_and_deserialize(solution), DataVector{5});
}

void test_copy_and_move() {
  gr::Solutions::GaugeWave<3> solution(0.25, 4.0);
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const auto created = TestHelpers::test_creation<gr::Solutions::GaugeWave<3>>(
      "Amplitude: 0.24\n"
      "Wavelength: 4.4");
  CHECK(created == gr::Solutions::GaugeWave<3>(0.24, 4.4));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.GaugeWave",
                  "[PointwiseFunctions][Unit]") {
  const double amplitude = 0.24;
  const double wavelength = 4.4;
  {
    const gr::Solutions::GaugeWave<1> solution(amplitude, wavelength);
    test_gauge_wave(
        solution, DataVector(5, std::numeric_limits<double>::signaling_NaN()));
    test_gauge_wave(solution, std::numeric_limits<double>::signaling_NaN());
  }
  {
    const gr::Solutions::GaugeWave<2> solution(amplitude, wavelength);
    test_gauge_wave(
        solution, DataVector(5, std::numeric_limits<double>::signaling_NaN()));
    test_gauge_wave(solution, std::numeric_limits<double>::signaling_NaN());
  }
  {
    const gr::Solutions::GaugeWave<3> solution(amplitude, wavelength);
    test_gauge_wave(
        solution, DataVector(5, std::numeric_limits<double>::signaling_NaN()));
    test_gauge_wave(solution, std::numeric_limits<double>::signaling_NaN());
  }
  test_consistency();
  test_copy_and_move();
  test_serialize();
  test_construct_from_options();

  CHECK_THROWS_WITH(
      []() { const gr::Solutions::GaugeWave<3> solution(-1.25, 1.0); }(),
      Catch::Matchers::ContainsSubstring("Amplitude must be less than one"));
  CHECK_THROWS_WITH(
      []() { const gr::Solutions::GaugeWave<3> solution(0.0, -0.25); }(),
      Catch::Matchers::ContainsSubstring("Wavelength must be non-negative"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<gr::Solutions::GaugeWave<3>>(
                        "Amplitude: -1.25\n"
                        "Wavelength: 1.0"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -1.25 is below the lower bound of -1"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<gr::Solutions::GaugeWave<3>>(
                        "Amplitude: 1.0\n"
                        "Wavelength: -0.25\n"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -0.25 is below the lower bound of 0"));
}
