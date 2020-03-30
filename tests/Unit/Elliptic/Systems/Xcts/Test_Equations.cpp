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
  {
    INFO("Symmetric result tensor");
    pypp::check_with_random_values<1>(
        &Xcts::longitudinal_shift<DataVector, Symmetry<1, 1>>, "Equations",
        {"longitudinal_shift"}, {{{-1., 1.}}}, used_for_size);
    pypp::check_with_random_values<1>(
        &Xcts::euclidean_longitudinal_shift<DataVector, Symmetry<1, 1>>,
        "Equations", {"euclidean_longitudinal_shift"}, {{{-1., 1.}}},
        used_for_size);
  }
  {
    INFO("Non-symmetric result tensor");
    pypp::check_with_random_values<1>(
        &Xcts::longitudinal_shift<DataVector, Symmetry<1, 2>>, "Equations",
        {"longitudinal_shift"}, {{{-1., 1.}}}, used_for_size);
    pypp::check_with_random_values<1>(
        &Xcts::euclidean_longitudinal_shift<DataVector, Symmetry<1, 2>>,
        "Equations", {"euclidean_longitudinal_shift"}, {{{-1., 1.}}},
        used_for_size);
  }
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
  CHECK_FOR_DATAVECTORS(
      test_computers,
      (Xcts::Equations::Hamiltonian, Xcts::Equations::HamiltonianAndLapse,
       Xcts::Equations::HamiltonianLapseAndShift),
      (Xcts::Geometry::Euclidean, Xcts::Geometry::NonEuclidean));
}
