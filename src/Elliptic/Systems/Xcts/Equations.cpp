// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Equations.hpp"

#include <array>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {

template <typename DataType>
void longitudinal_shift(
    const gsl::not_null<tnsr::II<DataType, 3>*> flux_for_shift,
    const tnsr::II<DataType, 3>& inv_conformal_metric,
    const tnsr::ii<DataType, 3>& shift_strain) noexcept {
  std::fill(flux_for_shift->begin(), flux_for_shift->end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          auto projection = 2. * (inv_conformal_metric.get(i, k) *
                                      inv_conformal_metric.get(j, l) -
                                  inv_conformal_metric.get(i, j) *
                                      inv_conformal_metric.get(k, l) / 3.);
          flux_for_shift->get(i, j) += projection * shift_strain.get(k, l);
        }
      }
    }
  }
}

template <typename DataType>
void euclidean_longitudinal_shift(
    const gsl::not_null<tnsr::II<DataType, 3>*> flux_for_shift,
    const tnsr::ii<DataType, 3>& shift_strain) noexcept {
  auto shift_strain_trace_term = get<0, 0>(shift_strain);
  for (size_t d = 1; d < 3; ++d) {
    shift_strain_trace_term += shift_strain.get(d, d);
  }
  shift_strain_trace_term *= 2. / 3.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
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

void add_distortion_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor) noexcept {
  get(*source_for_conformal_factor) -=
      0.03125 * pow<5>(get(conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_over_lapse_square);
}

void add_linearized_distortion_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  get(*source_for_conformal_factor) -=
      0.15625 * pow<4>(get(conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_over_lapse_square) *
      get(conformal_factor_correction);
}

void add_non_euclidean_hamiltonian_or_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*> source,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& field) noexcept {
  get(*source) += 0.125 * get(conformal_ricci_scalar) * get(field);
}

void lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  get(*source_for_lapse_times_conformal_factor) =
      get(lapse_times_conformal_factor) * pow<4>(get(conformal_factor)) *
          ((5. / 12.) * square(get(extrinsic_curvature_trace)) +
           2. * M_PI * (get(energy_density) + 2. * get(stress_trace))) +
      pow<5>(get(conformal_factor)) *
          (get(shift_dot_deriv_extrinsic_curvature_trace) -
           get(dt_extrinsic_curvature_trace));
}

void linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  get(*source_for_lapse_times_conformal_factor_correction) =
      (pow<4>(get(conformal_factor)) *
           get(lapse_times_conformal_factor_correction) +
       4. * get(lapse_times_conformal_factor) * pow<3>(get(conformal_factor)) *
           get(conformal_factor_correction)) *
          ((5. / 12.) * square(get(extrinsic_curvature_trace)) +
           2. * M_PI * (get(energy_density) + 2 * get(stress_trace))) +
      5. * pow<4>(get(conformal_factor)) * get(conformal_factor_correction) *
          (get(shift_dot_deriv_extrinsic_curvature_trace) -
           get(dt_extrinsic_curvature_trace));
}

void add_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    gsl::not_null<Scalar<DataVector>*> source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  get(*source_for_conformal_factor) -=
      0.03125 * pow<7>(get(conformal_factor)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
  get(*source_for_lapse_times_conformal_factor) +=
      0.21875 * pow<6>(get(conformal_factor)) /
      get(lapse_times_conformal_factor) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
}

void add_linearized_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    gsl::not_null<Scalar<DataVector>*> source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  get(*source_for_conformal_factor) -=
      0.03125 * pow<6>(get(conformal_factor)) *
      (7. * get(conformal_factor_correction) -
       2. * get(conformal_factor) / get(lapse_times_conformal_factor) *
           get(lapse_times_conformal_factor_correction)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
  get(*source_for_lapse_times_conformal_factor) +=
      0.21875 * pow<5>(get(conformal_factor)) *
      (6. * get(conformal_factor_correction) -
       get(conformal_factor) / get(lapse_times_conformal_factor) *
           get(lapse_times_conformal_factor_correction)) /
      get(lapse_times_conformal_factor) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
}

template <typename DataType>
void fully_contract(const gsl::not_null<Scalar<DataType>*> result,
                    const tnsr::II<DataType, 3>& tensor1,
                    const tnsr::II<DataType, 3>& tensor2) noexcept {
  get(*result) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) += tensor1.get(i, j) * tensor2.get(i, j);
    }
  }
}

template <typename DataType>
void fully_contract(const gsl::not_null<Scalar<DataType>*> result,
                    const tnsr::II<DataType, 3>& tensor1,
                    const tnsr::II<DataType, 3>& tensor2,
                    const tnsr::ii<DataType, 3>& metric) noexcept {
  get(*result) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          get(*result) += metric.get(i, k) * metric.get(j, l) *
                          tensor1.get(i, j) * tensor2.get(k, l);
        }
      }
    }
  }
}

template <Geometry ConformalGeometry>
void momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const std::optional<tnsr::ii<DataVector, 3>>& conformal_metric,
    const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    ASSERT(not conformal_metric.has_value() and
               not inv_conformal_metric.has_value(),
           "You don't need to pass a conformal metric to this function when it "
           "is specialized for a Euclidean conformal geometry.");
  } else {
    ASSERT(conformal_metric.has_value() and inv_conformal_metric.has_value(),
           "You must pass a conformal metric to this function when it is "
           "specialized for a non-Euclidean conformal geometry.");
  }
  auto longitudinal_shift_square = make_with_value<Scalar<DataVector>>(
      longitudinal_shift_minus_dt_conformal_metric, 0.);
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    fully_contract(make_not_null(&longitudinal_shift_square),
                   longitudinal_shift_minus_dt_conformal_metric,
                   longitudinal_shift_minus_dt_conformal_metric);
  } else {
    fully_contract(make_not_null(&longitudinal_shift_square),
                   longitudinal_shift_minus_dt_conformal_metric,
                   longitudinal_shift_minus_dt_conformal_metric,
                   *conformal_metric);
  }
  // Add shift terms to Hamiltonian and lapse equations
  get(*source_for_conformal_factor) -=
      0.03125 * pow<7>(get(conformal_factor)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_square);
  get(*source_for_lapse_times_conformal_factor) +=
      pow<6>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
      get(longitudinal_shift_square) * 7. / 32.;
  // Compute shift source
  // Begin with extrinsic curvature term
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    get<0>(*source_for_shift) = get<0>(extrinsic_curvature_trace_gradient);
    get<1>(*source_for_shift) = get<1>(extrinsic_curvature_trace_gradient);
    get<2>(*source_for_shift) = get<2>(extrinsic_curvature_trace_gradient);
  } else {
    raise_or_lower_index(source_for_shift, extrinsic_curvature_trace_gradient,
                         *inv_conformal_metric);
  }
  // Compute lapse deriv term to be contracted with longitudinal shift
  auto lapse_deriv_term =
      make_with_value<tnsr::I<DataVector, 3>>(conformal_factor_flux, 0.);
  for (size_t i = 0; i < 3; ++i) {
    lapse_deriv_term.get(i) =
        (lapse_times_conformal_factor_flux.get(i) /
             get(lapse_times_conformal_factor) -
         7. * conformal_factor_flux.get(i) / get(conformal_factor));
  }
  auto lapse_deriv_term_lo =
      make_with_value<tnsr::i<DataVector, 3>>(conformal_factor_flux, 0.);
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    get<0>(lapse_deriv_term_lo) = get<0>(lapse_deriv_term);
    get<1>(lapse_deriv_term_lo) = get<1>(lapse_deriv_term);
    get<2>(lapse_deriv_term_lo) = get<2>(lapse_deriv_term);
  } else {
    raise_or_lower_index(make_not_null(&lapse_deriv_term_lo), lapse_deriv_term,
                         *conformal_metric);
  }
  for (size_t i = 0; i < 3; ++i) {
    // Complete extrinsic curvature term
    source_for_shift->get(i) *=
        4. / 3. * get(lapse_times_conformal_factor) / get(conformal_factor);
    // Add momentum density term
    source_for_shift->get(i) += 16. * M_PI * get(lapse_times_conformal_factor) *
                                pow<3>(get(conformal_factor)) *
                                momentum_density.get(i);
    // Add longitudinal shift term
    for (size_t j = 0; j < 3; ++j) {
      source_for_shift->get(i) +=
          longitudinal_shift_minus_dt_conformal_metric.get(i, j) *
          lapse_deriv_term_lo.get(j);
    }
    source_for_shift->get(i) -= minus_div_dt_conformal_metric.get(i);
  }
}

template <Geometry ConformalGeometry>
void linearized_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift_correction,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const std::optional<tnsr::ii<DataVector, 3>>& conformal_metric,
    const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,
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
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    ASSERT(not conformal_metric.has_value() and
               not inv_conformal_metric.has_value(),
           "You don't need to pass a conformal metric to this function when it "
           "is specialized for a Euclidean conformal geometry.");
  } else {
    ASSERT(conformal_metric.has_value() and inv_conformal_metric.has_value(),
           "You must pass a conformal metric to this function when it is "
           "specialized for a non-Euclidean conformal geometry.");
  }
  auto longitudinal_shift_square = make_with_value<Scalar<DataVector>>(
      longitudinal_shift_minus_dt_conformal_metric, 0.);
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    fully_contract(make_not_null(&longitudinal_shift_square),
                   longitudinal_shift_minus_dt_conformal_metric,
                   longitudinal_shift_minus_dt_conformal_metric);
  } else {
    fully_contract(make_not_null(&longitudinal_shift_square),
                   longitudinal_shift_minus_dt_conformal_metric,
                   longitudinal_shift_minus_dt_conformal_metric,
                   *conformal_metric);
  }
  auto longitudinal_shift_dot_correction = make_with_value<Scalar<DataVector>>(
      longitudinal_shift_minus_dt_conformal_metric, 0.);
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    fully_contract(make_not_null(&longitudinal_shift_dot_correction),
                   longitudinal_shift_minus_dt_conformal_metric,
                   longitudinal_shift_correction);
  } else {
    fully_contract(make_not_null(&longitudinal_shift_dot_correction),
                   longitudinal_shift_minus_dt_conformal_metric,
                   longitudinal_shift_correction, *conformal_metric);
  }
  // Add shift terms to Hamiltonian and lapse equations
  get(*source_for_conformal_factor_correction) +=
      -7. / 32. * pow<6>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          get(longitudinal_shift_square) * get(conformal_factor_correction) +
      0.0625 * pow<7>(get(conformal_factor)) /
          pow<3>(get(lapse_times_conformal_factor)) *
          get(longitudinal_shift_square) *
          get(lapse_times_conformal_factor_correction) -
      0.0625 * pow<7>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          get(longitudinal_shift_dot_correction);
  get(*source_for_lapse_times_conformal_factor_correction) +=
      21. / 16. * pow<5>(get(conformal_factor)) /
          get(lapse_times_conformal_factor) * get(longitudinal_shift_square) *
          get(conformal_factor_correction) -
      7. / 32. * pow<6>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          get(longitudinal_shift_square) *
          get(lapse_times_conformal_factor_correction) +
      0.4375 * pow<6>(get(conformal_factor)) /
          get(lapse_times_conformal_factor) *
          get(longitudinal_shift_dot_correction) +
      // The conformal factor correction term for the shift.grad(K) term is
      // already added in `linearized_lapse_sources`
      pow<5>(get(conformal_factor)) *
          get(dot_product(shift_correction,
                          extrinsic_curvature_trace_gradient));
  // Compute shift source
  // Begin with extrinsic curvature term
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    get<0>(*source_for_shift_correction) =
        get<0>(extrinsic_curvature_trace_gradient);
    get<1>(*source_for_shift_correction) =
        get<1>(extrinsic_curvature_trace_gradient);
    get<2>(*source_for_shift_correction) =
        get<2>(extrinsic_curvature_trace_gradient);
  } else {
    raise_or_lower_index(source_for_shift_correction,
                         extrinsic_curvature_trace_gradient,
                         *inv_conformal_metric);
  }
  // Compute lapse deriv term to be contracted with longitudinal shift
  auto lapse_deriv_term =
      make_with_value<tnsr::I<DataVector, 3>>(conformal_factor_flux, 0.);
  auto lapse_deriv_correction_term =
      make_with_value<tnsr::I<DataVector, 3>>(conformal_factor_flux, 0.);
  for (size_t i = 0; i < 3; ++i) {
    lapse_deriv_term.get(i) =
        (lapse_times_conformal_factor_flux.get(i) /
             get(lapse_times_conformal_factor) -
         7. * conformal_factor_flux.get(i) / get(conformal_factor));
    lapse_deriv_correction_term.get(i) =
        (lapse_times_conformal_factor_flux_correction.get(i) /
             get(lapse_times_conformal_factor) -
         lapse_times_conformal_factor_flux.get(i) /
             square(get(lapse_times_conformal_factor)) *
             get(lapse_times_conformal_factor_correction) -
         7. * conformal_factor_flux_correction.get(i) / get(conformal_factor) +
         7. * conformal_factor_flux.get(i) / square(get(conformal_factor)) *
             get(conformal_factor_correction));
  }
  auto lapse_deriv_term_lo =
      make_with_value<tnsr::i<DataVector, 3>>(conformal_factor_flux, 0.);
  auto lapse_deriv_correction_term_lo =
      make_with_value<tnsr::i<DataVector, 3>>(conformal_factor_flux, 0.);
  if constexpr (ConformalGeometry == Geometry::Euclidean) {
    get<0>(lapse_deriv_term_lo) = get<0>(lapse_deriv_term);
    get<1>(lapse_deriv_term_lo) = get<1>(lapse_deriv_term);
    get<2>(lapse_deriv_term_lo) = get<2>(lapse_deriv_term);
    get<0>(lapse_deriv_correction_term_lo) =
        get<0>(lapse_deriv_correction_term);
    get<1>(lapse_deriv_correction_term_lo) =
        get<1>(lapse_deriv_correction_term);
    get<2>(lapse_deriv_correction_term_lo) =
        get<2>(lapse_deriv_correction_term);
  } else {
    raise_or_lower_index(make_not_null(&lapse_deriv_term_lo), lapse_deriv_term,
                         *conformal_metric);
    raise_or_lower_index(make_not_null(&lapse_deriv_correction_term_lo),
                         lapse_deriv_correction_term, *conformal_metric);
  }
  for (size_t i = 0; i < 3; ++i) {
    // Complete extrinsic curvature term
    source_for_shift_correction->get(i) *=
        4. / 3. *
        (get(lapse_times_conformal_factor_correction) / get(conformal_factor) -
         get(lapse_times_conformal_factor) / square(get(conformal_factor)) *
             get(conformal_factor_correction));
    // Add momentum density term
    source_for_shift_correction->get(i) +=
        16. * M_PI *
        (pow<3>(get(conformal_factor)) *
             get(lapse_times_conformal_factor_correction) +
         3. * square(get(conformal_factor)) *
             get(lapse_times_conformal_factor) *
             get(conformal_factor_correction)) *
        momentum_density.get(i);
    // Add longitudinal shift term
    for (size_t j = 0; j < 3; ++j) {
      source_for_shift_correction->get(i) +=
          longitudinal_shift_correction.get(i, j) * lapse_deriv_term_lo.get(j) +
          longitudinal_shift_minus_dt_conformal_metric.get(i, j) *
              lapse_deriv_correction_term_lo.get(j);
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEOM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DTYPE(_, data)                                   \
  template void fully_contract(                                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,              \
      const tnsr::II<DTYPE(data), 3>& tensor1,                       \
      const tnsr::II<DTYPE(data), 3>& tensor2) noexcept;             \
  template void fully_contract(                                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,              \
      const tnsr::II<DTYPE(data), 3>& tensor1,                       \
      const tnsr::II<DTYPE(data), 3>& tensor2,                       \
      const tnsr::ii<DTYPE(data), 3>& metric) noexcept;              \
  template void longitudinal_shift(                                  \
      const gsl::not_null<tnsr::II<DTYPE(data), 3>*> flux_for_shift, \
      const tnsr::II<DTYPE(data), 3>& inv_conformal_metric,          \
      const tnsr::ii<DTYPE(data), 3>& shift_strain) noexcept;        \
  template void euclidean_longitudinal_shift(                        \
      const gsl::not_null<tnsr::II<DTYPE(data), 3>*> flux_for_shift, \
      const tnsr::ii<DTYPE(data), 3>& shift_strain) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE, (double, DataVector))

#define INSTANTIATE(_, data)                                              \
  template void momentum_sources<GEOM(data)>(                             \
      gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,     \
      gsl::not_null<Scalar<DataVector>*>                                  \
          source_for_lapse_times_conformal_factor,                        \
      gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift,            \
      const tnsr::I<DataVector, 3>& momentum_density,                     \
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,   \
      const std::optional<tnsr::ii<DataVector, 3>>& conformal_metric,     \
      const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric, \
      const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,        \
      const Scalar<DataVector>& conformal_factor,                         \
      const Scalar<DataVector>& lapse_times_conformal_factor,             \
      const tnsr::I<DataVector, 3>& conformal_factor_flux,                \
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,    \
      const tnsr::II<DataVector, 3>&                                      \
          longitudinal_shift_minus_dt_conformal_metric) noexcept;         \
  template void linearized_momentum_sources<GEOM(data)>(                  \
      gsl::not_null<Scalar<DataVector>*>                                  \
          source_for_conformal_factor_correction,                         \
      gsl::not_null<Scalar<DataVector>*>                                  \
          source_for_lapse_times_conformal_factor_correction,             \
      gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift_correction, \
      const tnsr::I<DataVector, 3>& momentum_density,                     \
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,   \
      const std::optional<tnsr::ii<DataVector, 3>>& conformal_metric,     \
      const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric, \
      const Scalar<DataVector>& conformal_factor,                         \
      const Scalar<DataVector>& lapse_times_conformal_factor,             \
      const tnsr::I<DataVector, 3>& conformal_factor_flux,                \
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,    \
      const tnsr::II<DataVector, 3>&                                      \
          longitudinal_shift_minus_dt_conformal_metric,                   \
      const Scalar<DataVector>& conformal_factor_correction,              \
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,  \
      const tnsr::I<DataVector, 3>& shift_correction,                     \
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,     \
      const tnsr::I<DataVector, 3>&                                       \
          lapse_times_conformal_factor_flux_correction,                   \
      const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Geometry::Euclidean, Geometry::NonEuclidean))

#undef INSTANTIATE
#undef INSTANTIATE_DTYPE
#undef GEOM
#undef DTYPE

}  // namespace Xcts
