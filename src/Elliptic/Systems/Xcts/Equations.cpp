// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {

void longitudinal_shift(
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_for_shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain) noexcept {
  auto shift_strain_trace_term = get<0, 0>(shift_strain);
  for (size_t d = 1; d < 3; d++) {
    shift_strain_trace_term += shift_strain.get(d, d);
  }
  shift_strain_trace_term *= 2. / 3.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      flux_for_shift->get(i, j) = 2. * shift_strain.get(i, j);
    }
    flux_for_shift->get(i, i) -= shift_strain_trace_term;
  }
}

void hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor) noexcept {
  get(*source_for_conformal_factor) =
      (square(get(extrinsic_curvature_trace)) / 12. -
       2. * M_PI * get(energy_density)) *
      pow<5>(get(conformal_factor));
}

void linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  get(*source_for_conformal_factor_correction) =
      ((5. / 12.) * square(get(extrinsic_curvature_trace)) -
       10. * M_PI * get(energy_density)) *
      pow<4>(get(conformal_factor)) * get(conformal_factor_correction);
}

void lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  get(*source_for_lapse_times_conformal_factor) =
      ((5. / 12.) * square(get(extrinsic_curvature_trace)) +
       2. * M_PI * (get(energy_density) + 2. * get(stress_trace))) *
      get(lapse_times_conformal_factor) * pow<4>(get(conformal_factor));
}

void linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  get(*source_for_lapse_times_conformal_factor_correction) =
      ((5. / 12.) * square(get(extrinsic_curvature_trace)) +
       2. * M_PI * (get(energy_density) + 2 * get(stress_trace))) *
      (pow<4>(get(conformal_factor)) *
           get(lapse_times_conformal_factor_correction) +
       4. * get(lapse_times_conformal_factor) * pow<3>(get(conformal_factor)) *
           get(conformal_factor_correction));
}

void momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_for_shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain) noexcept {
  auto longitudinal_shift =
      make_with_value<tnsr::IJ<DataVector, 3, Frame::Inertial>>(shift_strain,
                                                                0.);
  Xcts::longitudinal_shift(make_not_null(&longitudinal_shift), shift_strain);
  // Add shift terms to Hamiltonian and lapse equations
  const DataVector shift_strain_traceless_square =
      (get(pointwise_l2_norm_square(shift_strain)) -
       square(get<0, 0>(shift_strain) + get<1, 1>(shift_strain) +
              get<2, 2>(shift_strain)) /
           3.);
  get(*source_for_conformal_factor) -=
      pow<7>(get(conformal_factor)) /
      square(get(lapse_times_conformal_factor)) *
      shift_strain_traceless_square / 8.;
  get(*source_for_lapse_times_conformal_factor) +=
      pow<6>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
          shift_strain_traceless_square * 7. / 8. +
      pow<5>(get(conformal_factor)) *
          get(dot_product(shift, extrinsic_curvature_trace_gradient));
  // Compute shift source
  std::fill(source_for_shift->begin(), source_for_shift->end(), 0.);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      source_for_shift->get(i) +=
          longitudinal_shift.get(i, j) *
          (lapse_times_conformal_factor_gradient.get(j) /
               get(lapse_times_conformal_factor) -
           7. * conformal_factor_gradient.get(j) / get(conformal_factor));
    }
    source_for_shift->get(i) +=
        16. * M_PI * get(lapse_times_conformal_factor) *
            pow<3>(get(conformal_factor)) * momentum_density.get(i) +
        4. / 3. * get(lapse_times_conformal_factor) / get(conformal_factor) *
            extrinsic_curvature_trace_gradient.get(i);
  }
}

void linearized_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_for_shift_correction,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift_correction,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        conformal_factor_gradient_correction,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient_correction,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        shift_strain_correction) noexcept {
  auto longitudinal_shift =
      make_with_value<tnsr::IJ<DataVector, 3, Frame::Inertial>>(shift_strain,
                                                                0.);
  Xcts::longitudinal_shift(make_not_null(&longitudinal_shift), shift_strain);
  auto longitudinal_shift_correction =
      make_with_value<tnsr::IJ<DataVector, 3, Frame::Inertial>>(
          shift_strain_correction, 0.);
  Xcts::longitudinal_shift(make_not_null(&longitudinal_shift_correction),
                           shift_strain_correction);
  // Add shift terms to Hamiltonian and lapse equations
  const DataVector shift_strain_trace = get<0, 0>(shift_strain) +
                                        get<1, 1>(shift_strain) +
                                        get<2, 2>(shift_strain);
  const DataVector shift_strain_traceless_square =
      (get(pointwise_l2_norm_square(shift_strain)) -
       square(shift_strain_trace) / 3.);
  DataVector shift_strain_traceless_square_correction{
      conformal_factor.begin()->size(), 0.};
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      shift_strain_traceless_square_correction +=
          shift_strain.get(i, j) * shift_strain_correction.get(i, j);
    }
  }
  shift_strain_traceless_square_correction -=
      shift_strain_trace *
      (get<0, 0>(shift_strain_correction) + get<1, 1>(shift_strain_correction) +
       get<2, 2>(shift_strain_correction)) /
      3.;
  get(*source_for_conformal_factor_correction) +=
      -7. / 8. * pow<6>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          shift_strain_traceless_square * get(conformal_factor_correction) +
      0.25 * pow<7>(get(conformal_factor)) /
          pow<3>(get(lapse_times_conformal_factor)) *
          shift_strain_traceless_square *
          get(lapse_times_conformal_factor_correction) -
      0.25 * pow<7>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          shift_strain_traceless_square_correction;
  get(*source_for_lapse_times_conformal_factor_correction) +=
      21. / 4. * pow<5>(get(conformal_factor)) /
          get(lapse_times_conformal_factor) * shift_strain_traceless_square *
          get(conformal_factor_correction) -
      7. / 8. * pow<6>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          shift_strain_traceless_square *
          get(lapse_times_conformal_factor_correction) +
      1.75 * pow<6>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
          shift_strain_traceless_square_correction +
      5. * pow<4>(get(conformal_factor)) *
          get(dot_product(shift, extrinsic_curvature_trace_gradient)) *
          get(conformal_factor_correction) +
      pow<5>(get(conformal_factor)) *
          get(dot_product(shift_correction,
                          extrinsic_curvature_trace_gradient));
  // Compute shift source
  std::fill(source_for_shift_correction->begin(),
            source_for_shift_correction->end(), 0.);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      source_for_shift_correction->get(i) +=
          longitudinal_shift_correction.get(i, j) *
              (lapse_times_conformal_factor_gradient.get(j) /
                   get(lapse_times_conformal_factor) -
               7. * conformal_factor_gradient.get(j) / get(conformal_factor)) +
          longitudinal_shift.get(i, j) *
              (lapse_times_conformal_factor_gradient_correction.get(j) /
                   get(lapse_times_conformal_factor) -
               lapse_times_conformal_factor_gradient.get(j) /
                   square(get(lapse_times_conformal_factor)) *
                   get(lapse_times_conformal_factor_correction) -
               7. * conformal_factor_gradient_correction.get(j) /
                   get(conformal_factor) +
               7. * conformal_factor_gradient.get(j) /
                   square(get(conformal_factor)) *
                   get(conformal_factor_correction));
    }
    source_for_shift_correction->get(i) +=
        16. * M_PI *
            (pow<3>(get(conformal_factor)) *
                 get(lapse_times_conformal_factor_correction) +
             3. * square(get(conformal_factor)) *
                 get(lapse_times_conformal_factor) *
                 get(conformal_factor_correction)) *
            momentum_density.get(i) +
        4. / 3. *
            (get(lapse_times_conformal_factor_correction) /
                 get(conformal_factor) -
             get(lapse_times_conformal_factor) / square(get(conformal_factor)) *
                 get(conformal_factor_correction)) *
            extrinsic_curvature_trace_gradient.get(i);
  }
}

}  // namespace Xcts
