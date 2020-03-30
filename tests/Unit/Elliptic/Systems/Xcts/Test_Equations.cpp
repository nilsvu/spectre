// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Elliptic/FirstOrderSystem.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// Wrappers to translate between Python and C++ functions

void add_distortion_hamiltonian_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor) noexcept {
  get(*source_for_conformal_factor) = 0.;
  Xcts::add_distortion_hamiltonian_sources(
      source_for_conformal_factor,
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      conformal_factor);
}

void add_linearized_distortion_hamiltonian_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  get(*source_for_conformal_factor) = 0.;
  Xcts::add_linearized_distortion_hamiltonian_sources(
      source_for_conformal_factor,
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      conformal_factor, conformal_factor_correction);
}

void add_non_euclidean_hamiltonian_or_lapse_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*> source,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& field) noexcept {
  get(*source) = 0.;
  Xcts::add_non_euclidean_hamiltonian_or_lapse_sources(
      source, conformal_ricci_scalar, field);
}

void add_distortion_hamiltonian_and_lapse_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  get(*source_for_conformal_factor) = 0.;
  get(*source_for_lapse_times_conformal_factor) = 0.;
  Xcts::add_distortion_hamiltonian_and_lapse_sources(
      source_for_conformal_factor, source_for_lapse_times_conformal_factor,
      longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
      lapse_times_conformal_factor);
}

void add_linearized_distortion_hamiltonian_and_lapse_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  get(*source_for_conformal_factor_correction) = 0.;
  get(*source_for_lapse_times_conformal_factor_correction) = 0.;
  Xcts::add_linearized_distortion_hamiltonian_and_lapse_sources(
      source_for_conformal_factor_correction,
      source_for_lapse_times_conformal_factor_correction,
      longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
}

void euclidean_momentum_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  get(*source_for_conformal_factor) = 0.;
  get(*source_for_lapse_times_conformal_factor) = 0.;
  Xcts::momentum_sources<Xcts::Geometry::Euclidean>(
      source_for_conformal_factor, source_for_lapse_times_conformal_factor,
      source_for_shift, momentum_density, extrinsic_curvature_trace_gradient,
      std::nullopt, std::nullopt, minus_div_dt_conformal_metric,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
}

void non_euclidean_momentum_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  get(*source_for_conformal_factor) = 0.;
  get(*source_for_lapse_times_conformal_factor) = 0.;
  Xcts::momentum_sources<Xcts::Geometry::NonEuclidean>(
      source_for_conformal_factor, source_for_lapse_times_conformal_factor,
      source_for_shift, momentum_density, extrinsic_curvature_trace_gradient,
      conformal_metric, inv_conformal_metric, minus_div_dt_conformal_metric,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
}

void euclidean_linearized_momentum_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift_correction,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept {
  get(*source_for_conformal_factor_correction) = 0.;
  get(*source_for_lapse_times_conformal_factor_correction) = 0.;
  Xcts::linearized_momentum_sources<Xcts::Geometry::Euclidean>(
      source_for_conformal_factor_correction,
      source_for_lapse_times_conformal_factor_correction,
      source_for_shift_correction, momentum_density,
      extrinsic_curvature_trace_gradient, std::nullopt, std::nullopt,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
}

void non_euclidean_linearized_momentum_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift_correction,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept {
  get(*source_for_conformal_factor_correction) = 0.;
  get(*source_for_lapse_times_conformal_factor_correction) = 0.;
  Xcts::linearized_momentum_sources<Xcts::Geometry::NonEuclidean>(
      source_for_conformal_factor_correction,
      source_for_lapse_times_conformal_factor_correction,
      source_for_shift_correction, momentum_density,
      extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, conformal_factor, lapse_times_conformal_factor,
      conformal_factor_flux, lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
}

void test_equations(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(&Xcts::longitudinal_shift<DataVector>,
                                    "Equations", {"longitudinal_shift"},
                                    {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &Xcts::euclidean_longitudinal_shift<DataVector>, "Equations",
      {"euclidean_longitudinal_shift"}, {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(&Xcts::hamiltonian_sources, "Equations",
                                    {"hamiltonian_sources"}, {{{-1., 1.}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(
      &Xcts::linearized_hamiltonian_sources, "Equations",
      {"linearized_hamiltonian_sources"}, {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &add_distortion_hamiltonian_sources_wrapper, "Equations",
      {"distortion_hamiltonian_sources"}, {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &add_linearized_distortion_hamiltonian_sources_wrapper, "Equations",
      {"linearized_distortion_hamiltonian_sources"}, {{{-1., 1.}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      &add_non_euclidean_hamiltonian_or_lapse_sources_wrapper, "Equations",
      {"non_euclidean_hamiltonian_or_lapse_sources"}, {{{-1., 1.}}},
      used_for_size);
  pypp::check_with_random_values<1>(&Xcts::lapse_sources, "Equations",
                                    {"lapse_sources"}, {{{-1., 1.}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(&Xcts::linearized_lapse_sources,
                                    "Equations", {"linearized_lapse_sources"},
                                    {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &add_distortion_hamiltonian_and_lapse_sources_wrapper, "Equations",
      {"distortion_hamiltonian_sources_with_lapse", "distortion_lapse_sources"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &add_linearized_distortion_hamiltonian_and_lapse_sources_wrapper,
      "Equations",
      {"linearized_distortion_hamiltonian_sources_with_lapse",
       "linearized_distortion_lapse_sources"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &euclidean_momentum_sources_wrapper, "Equations",
      {"euclidean_distortion_hamiltonian_sources_with_lapse_and_shift",
       "euclidean_distortion_lapse_sources_with_shift",
       "euclidean_momentum_sources"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &non_euclidean_momentum_sources_wrapper, "Equations",
      {"distortion_hamiltonian_sources_with_lapse_and_shift",
       "distortion_lapse_sources_with_shift", "momentum_sources"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &euclidean_linearized_momentum_sources_wrapper, "Equations",
      {"euclidean_linearized_distortion_hamiltonian_sources_with_lapse_and_"
       "shift",
       "euclidean_linearized_distortion_lapse_sources_with_shift",
       "euclidean_linearized_momentum_sources"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &non_euclidean_linearized_momentum_sources_wrapper, "Equations",
      {"linearized_distortion_hamiltonian_sources_with_lapse_and_shift",
       "linearized_distortion_lapse_sources_with_shift",
       "linearized_momentum_sources"},
      {{{-1., 1.}}}, used_for_size);
}

void test_linearization(const DataVector used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist_positive(0., 1.);
  std::uniform_real_distribution<> dist_factor(0.5, 2.);
  std::uniform_real_distribution<> dist_isotropic(-1., 1.);
  // The linearization is correct to this order
  const double eps = 1.e-3;
  std::uniform_real_distribution<> dist_eps(-eps, eps);
  Approx custom_approx = Approx::custom().epsilon(eps).scale(1.);

  // Background fields
  const auto extrinsic_curvature_trace =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&generator),
                                                  make_not_null(&dist_positive),
                                                  used_for_size);
  const auto dt_extrinsic_curvature_trace =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto extrinsic_curvature_trace_gradient =
      make_with_random_values<tnsr::i<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto longitudinal_shift_minus_dt_conformal_metric_square =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&generator),
                                                  make_not_null(&dist_positive),
                                                  used_for_size);
  const auto longitudinal_shift_minus_dt_conformal_metric_over_lapse_square =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&generator),
                                                  make_not_null(&dist_positive),
                                                  used_for_size);
  const auto minus_div_dt_conformal_metric =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto inv_conformal_metric =
      make_with_random_values<tnsr::II<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto energy_density = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist_positive), used_for_size);
  const auto stress_trace = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist_positive), used_for_size);
  const auto momentum_density = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist_positive), used_for_size);
  const auto shift_dot_deriv_extrinsic_curvature_trace =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);

  // Variables for linearized Hamiltonian sources
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist_factor), used_for_size);
  const auto conformal_factor_correction =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  const Scalar<DataVector> conformal_factor_corrected{
      get(conformal_factor) + get(conformal_factor_correction)};

  // Test linearized Hamiltonian sources
  Scalar<DataVector> hamiltonian_source{used_for_size.size()};
  Xcts::hamiltonian_sources(make_not_null(&hamiltonian_source), energy_density,
                            extrinsic_curvature_trace, conformal_factor);
  Xcts::add_distortion_hamiltonian_sources(
      make_not_null(&hamiltonian_source),
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      conformal_factor);
  Scalar<DataVector> hamiltonian_source_corrected{used_for_size.size()};
  Xcts::hamiltonian_sources(make_not_null(&hamiltonian_source_corrected),
                            energy_density, extrinsic_curvature_trace,
                            conformal_factor_corrected);
  Xcts::add_distortion_hamiltonian_sources(
      make_not_null(&hamiltonian_source_corrected),
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      conformal_factor_corrected);
  DataVector hamiltonian_diff =
      get(hamiltonian_source_corrected) - get(hamiltonian_source);
  Scalar<DataVector> hamiltonian_diff_linear{used_for_size.size()};
  Xcts::linearized_hamiltonian_sources(
      make_not_null(&hamiltonian_diff_linear), energy_density,
      extrinsic_curvature_trace, conformal_factor, conformal_factor_correction);
  Xcts::add_linearized_distortion_hamiltonian_sources(
      make_not_null(&hamiltonian_diff_linear),
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      conformal_factor, conformal_factor_correction);
  CHECK_ITERABLE_CUSTOM_APPROX(hamiltonian_diff, get(hamiltonian_diff_linear),
                               custom_approx);

  // Variables for linearized lapse sources
  const auto lapse_times_conformal_factor =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&generator),
                                                  make_not_null(&dist_factor),
                                                  used_for_size);
  const auto lapse_times_conformal_factor_correction =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  const Scalar<DataVector> lapse_times_conformal_factor_corrected{
      get(lapse_times_conformal_factor) +
      get(lapse_times_conformal_factor_correction)};

  // Test linearized lapse sources
  Scalar<DataVector> lapse_source{used_for_size.size()};
  get(hamiltonian_source) = 0.;
  Xcts::lapse_sources(make_not_null(&lapse_source), energy_density,
                      stress_trace, extrinsic_curvature_trace,
                      dt_extrinsic_curvature_trace,
                      shift_dot_deriv_extrinsic_curvature_trace,
                      conformal_factor, lapse_times_conformal_factor);
  Xcts::add_distortion_hamiltonian_and_lapse_sources(
      make_not_null(&hamiltonian_source), make_not_null(&lapse_source),
      longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
      lapse_times_conformal_factor);
  Scalar<DataVector> lapse_source_corrected{used_for_size.size()};
  get(hamiltonian_source_corrected) = 0.;
  Xcts::lapse_sources(
      make_not_null(&lapse_source_corrected), energy_density, stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      shift_dot_deriv_extrinsic_curvature_trace, conformal_factor_corrected,
      lapse_times_conformal_factor_corrected);
  Xcts::add_distortion_hamiltonian_and_lapse_sources(
      make_not_null(&hamiltonian_source_corrected),
      make_not_null(&lapse_source_corrected),
      longitudinal_shift_minus_dt_conformal_metric_square,
      conformal_factor_corrected, lapse_times_conformal_factor_corrected);
  DataVector lapse_diff = get(lapse_source_corrected) - get(lapse_source);
  hamiltonian_diff =
      get(hamiltonian_source_corrected) - get(hamiltonian_source);
  Scalar<DataVector> lapse_diff_linear{used_for_size.size()};
  get(hamiltonian_diff_linear) = 0.;
  Xcts::linearized_lapse_sources(
      make_not_null(&lapse_diff_linear), energy_density, stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
  Xcts::add_linearized_distortion_hamiltonian_and_lapse_sources(
      make_not_null(&hamiltonian_diff_linear),
      make_not_null(&lapse_diff_linear),
      longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
  CHECK_ITERABLE_CUSTOM_APPROX(lapse_diff, get(lapse_diff_linear),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(hamiltonian_diff, get(hamiltonian_diff_linear),
                               custom_approx);

  // Variables for linearized momentum sources
  const auto shift = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist_isotropic), used_for_size);
  const auto shift_correction = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  tnsr::I<DataVector, 3> shift_corrected{used_for_size.size()};
  for (size_t i = 0; i < 3; ++i) {
    shift_corrected.get(i) = shift.get(i) + shift_correction.get(i);
  }
  const auto conformal_factor_flux =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto conformal_factor_flux_correction =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  tnsr::I<DataVector, 3> conformal_factor_flux_corrected{used_for_size.size()};
  for (size_t i = 0; i < 3; ++i) {
    conformal_factor_flux_corrected.get(i) =
        conformal_factor_flux.get(i) + conformal_factor_flux_correction.get(i);
  }
  const auto lapse_times_conformal_factor_flux =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto lapse_times_conformal_factor_flux_correction =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  tnsr::I<DataVector, 3> lapse_times_conformal_factor_flux_corrected{
      used_for_size.size()};
  for (size_t i = 0; i < 3; ++i) {
    lapse_times_conformal_factor_flux_corrected.get(i) =
        lapse_times_conformal_factor_flux.get(i) +
        lapse_times_conformal_factor_flux_correction.get(i);
  }
  const auto longitudinal_shift =
      make_with_random_values<tnsr::II<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_isotropic),
          used_for_size);
  const auto longitudinal_shift_correction =
      make_with_random_values<tnsr::II<DataVector, 3>>(
          make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  tnsr::II<DataVector, 3> longitudinal_shift_corrected{used_for_size.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      longitudinal_shift_corrected.get(i, j) =
          longitudinal_shift.get(i, j) +
          longitudinal_shift_correction.get(i, j);
    }
  }

  // Test linearized momentum sources
  // (i) Euclidean conformal metric
  tnsr::I<DataVector, 3> momentum_source{used_for_size.size()};
  get(hamiltonian_source) = 0.;
  Xcts::lapse_sources(make_not_null(&lapse_source), energy_density,
                      stress_trace, extrinsic_curvature_trace,
                      dt_extrinsic_curvature_trace,
                      dot_product(shift, extrinsic_curvature_trace_gradient),
                      conformal_factor, lapse_times_conformal_factor);
  Xcts::momentum_sources<Xcts::Geometry::Euclidean>(
      make_not_null(&hamiltonian_source), make_not_null(&lapse_source),
      make_not_null(&momentum_source), momentum_density,
      extrinsic_curvature_trace_gradient, std::nullopt, std::nullopt,
      minus_div_dt_conformal_metric, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux, longitudinal_shift);
  tnsr::I<DataVector, 3> momentum_source_corrected{used_for_size.size()};
  get(hamiltonian_source_corrected) = 0.;
  Xcts::lapse_sources(
      make_not_null(&lapse_source_corrected), energy_density, stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      dot_product(shift_corrected, extrinsic_curvature_trace_gradient),
      conformal_factor_corrected, lapse_times_conformal_factor_corrected);
  Xcts::momentum_sources<Xcts::Geometry::Euclidean>(
      make_not_null(&hamiltonian_source_corrected),
      make_not_null(&lapse_source_corrected),
      make_not_null(&momentum_source_corrected), momentum_density,
      extrinsic_curvature_trace_gradient, std::nullopt, std::nullopt,
      minus_div_dt_conformal_metric, conformal_factor_corrected,
      lapse_times_conformal_factor_corrected, conformal_factor_flux_corrected,
      lapse_times_conformal_factor_flux_corrected,
      longitudinal_shift_corrected);
  hamiltonian_diff =
      get(hamiltonian_source_corrected) - get(hamiltonian_source);
  lapse_diff = get(lapse_source_corrected) - get(lapse_source);
  tnsr::I<DataVector, 3> momentum_diff{used_for_size.size()};
  for (size_t i = 0; i < 3; ++i) {
    momentum_diff.get(i) =
        momentum_source_corrected.get(i) - momentum_source.get(i);
  }
  tnsr::I<DataVector, 3> momentum_diff_linear{used_for_size.size()};
  get(hamiltonian_diff_linear) = 0.;
  Xcts::linearized_lapse_sources(
      make_not_null(&lapse_diff_linear), energy_density, stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      dot_product(shift, extrinsic_curvature_trace_gradient), conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
  Xcts::linearized_momentum_sources<Xcts::Geometry::Euclidean>(
      make_not_null(&hamiltonian_diff_linear),
      make_not_null(&lapse_diff_linear), make_not_null(&momentum_diff_linear),
      momentum_density, extrinsic_curvature_trace_gradient, std::nullopt,
      std::nullopt, conformal_factor, lapse_times_conformal_factor,
      conformal_factor_flux, lapse_times_conformal_factor_flux,
      longitudinal_shift, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
  CHECK_ITERABLE_CUSTOM_APPROX(hamiltonian_diff, get(hamiltonian_diff_linear),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(lapse_diff, get(lapse_diff_linear),
                               custom_approx);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_CUSTOM_APPROX(momentum_diff.get(i),
                                 momentum_diff_linear.get(i), custom_approx);
  }
  // (ii) Non-Euclidean conformal metric
  get(hamiltonian_source) = 0.;
  Xcts::lapse_sources(make_not_null(&lapse_source), energy_density,
                      stress_trace, extrinsic_curvature_trace,
                      dt_extrinsic_curvature_trace,
                      dot_product(shift, extrinsic_curvature_trace_gradient),
                      conformal_factor, lapse_times_conformal_factor);
  Xcts::momentum_sources<Xcts::Geometry::NonEuclidean>(
      make_not_null(&hamiltonian_source), make_not_null(&lapse_source),
      make_not_null(&momentum_source), momentum_density,
      extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, minus_div_dt_conformal_metric, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux, longitudinal_shift);
  get(hamiltonian_source_corrected) = 0.;
  Xcts::lapse_sources(
      make_not_null(&lapse_source_corrected), energy_density, stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      dot_product(shift_corrected, extrinsic_curvature_trace_gradient),
      conformal_factor_corrected, lapse_times_conformal_factor_corrected);
  Xcts::momentum_sources<Xcts::Geometry::NonEuclidean>(
      make_not_null(&hamiltonian_source_corrected),
      make_not_null(&lapse_source_corrected),
      make_not_null(&momentum_source_corrected), momentum_density,
      extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, minus_div_dt_conformal_metric,
      conformal_factor_corrected, lapse_times_conformal_factor_corrected,
      conformal_factor_flux_corrected,
      lapse_times_conformal_factor_flux_corrected,
      longitudinal_shift_corrected);
  hamiltonian_diff =
      get(hamiltonian_source_corrected) - get(hamiltonian_source);
  lapse_diff = get(lapse_source_corrected) - get(lapse_source);
  for (size_t i = 0; i < 3; ++i) {
    momentum_diff.get(i) =
        momentum_source_corrected.get(i) - momentum_source.get(i);
  }
  get(hamiltonian_diff_linear) = 0.;
  Xcts::linearized_lapse_sources(
      make_not_null(&lapse_diff_linear), energy_density, stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      dot_product(shift, extrinsic_curvature_trace_gradient), conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
  Xcts::linearized_momentum_sources<Xcts::Geometry::NonEuclidean>(
      make_not_null(&hamiltonian_diff_linear),
      make_not_null(&lapse_diff_linear), make_not_null(&momentum_diff_linear),
      momentum_density, extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, conformal_factor, lapse_times_conformal_factor,
      conformal_factor_flux, lapse_times_conformal_factor_flux,
      longitudinal_shift, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
  CHECK_ITERABLE_CUSTOM_APPROX(hamiltonian_diff, get(hamiltonian_diff_linear),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(lapse_diff, get(lapse_diff_linear),
                               custom_approx);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_CUSTOM_APPROX(momentum_diff.get(i),
                                 momentum_diff_linear.get(i), custom_approx);
  }
}

template <Xcts::Equations EnabledEquations, Xcts::Geometry ConformalGeometry>
void test_computers(const DataVector& used_for_size) {
  CAPTURE(EnabledEquations);
  CAPTURE(ConformalGeometry);
  using system = Xcts::FirstOrderSystem<EnabledEquations, ConformalGeometry>;
  TestHelpers::elliptic::test_first_order_fluxes_computer<system>(
      typename system::fluxes{}, used_for_size);
  TestHelpers::elliptic::test_first_order_sources_computer<system>(
      used_for_size);
  using linearized_system = typename system::linearized_system;
  TestHelpers::elliptic::test_first_order_fluxes_computer<linearized_system>(
      typename linearized_system::fluxes{}, used_for_size);
  TestHelpers::elliptic::test_first_order_sources_computer<linearized_system>(
      used_for_size);
  TestHelpers::elliptic::test_linearization<system>(1.e-3, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Elliptic/Systems/Xcts"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  test_equations(dv);
  test_linearization(dv);
  CHECK_FOR_DATAVECTORS(
      test_computers,
      (Xcts::Equations::Hamiltonian, Xcts::Equations::HamiltonianAndLapse,
       Xcts::Equations::HamiltonianLapseAndShift),
      (Xcts::Geometry::Euclidean, Xcts::Geometry::NonEuclidean));
}
