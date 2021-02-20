// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Equations.hpp"

#include <algorithm>
#include <functional>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {

void add_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor) noexcept {
  get(*hamiltonian_constraint) +=
      (square(get(extrinsic_curvature_trace)) / 12. -
       2. * M_PI * get(energy_density)) *
      pow<5>(get(conformal_factor));
}

void add_linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  get(*linearized_hamiltonian_constraint) +=
      ((5. / 12.) * square(get(extrinsic_curvature_trace)) -
       10. * M_PI * get(energy_density)) *
      pow<4>(get(conformal_factor)) * get(conformal_factor_correction);
}

void add_distortion_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor) noexcept {
  get(*hamiltonian_constraint) -=
      0.03125 * pow<5>(get(conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_over_lapse_square);
}

void add_linearized_distortion_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  get(*linearized_hamiltonian_constraint) -=
      0.15625 * pow<4>(get(conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_over_lapse_square) *
      get(conformal_factor_correction);
}

void add_curved_hamiltonian_or_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_or_lapse_equation,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& field) noexcept {
  get(*hamiltonian_or_lapse_equation) +=
      0.125 * get(conformal_ricci_scalar) * get(field);
}

void add_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  get(*lapse_equation) +=
      get(lapse_times_conformal_factor) * pow<4>(get(conformal_factor)) *
          ((5. / 12.) * square(get(extrinsic_curvature_trace)) +
           2. * M_PI * (get(energy_density) + 2. * get(stress_trace))) +
      pow<5>(get(conformal_factor)) *
          (get(shift_dot_deriv_extrinsic_curvature_trace) -
           get(dt_extrinsic_curvature_trace));
}

void add_linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
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
  get(*linearized_lapse_equation) +=
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
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  get(*hamiltonian_constraint) -=
      0.03125 * pow<7>(get(conformal_factor)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
  get(*lapse_equation) +=
      0.21875 * pow<6>(get(conformal_factor)) /
      get(lapse_times_conformal_factor) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
}

void add_linearized_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  get(*hamiltonian_constraint) -=
      0.03125 * pow<6>(get(conformal_factor)) *
      (7. * get(conformal_factor_correction) -
       2. * get(conformal_factor) / get(lapse_times_conformal_factor) *
           get(lapse_times_conformal_factor_correction)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
  get(*lapse_equation) +=
      0.21875 * pow<5>(get(conformal_factor)) *
      (6. * get(conformal_factor_correction) -
       get(conformal_factor) / get(lapse_times_conformal_factor) *
           get(lapse_times_conformal_factor_correction)) /
      get(lapse_times_conformal_factor) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
}

namespace detail {
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
}  // namespace detail

template <Geometry ConformalGeometry>
void add_momentum_sources_impl(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const std::optional<std::reference_wrapper<const tnsr::ii<DataVector, 3>>>
        conformal_metric,
    const std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    ASSERT(not conformal_metric.has_value() and
               not inv_conformal_metric.has_value(),
           "You don't need to pass a conformal metric to this function when it "
           "is specialized for a flat conformal geometry in Cartesian "
           "coordinates.");
  } else {
    ASSERT(conformal_metric.has_value() and inv_conformal_metric.has_value(),
           "You must pass a conformal metric to this function when it is "
           "specialized for a curved conformal geometry.");
  }
  auto longitudinal_shift_square = make_with_value<Scalar<DataVector>>(
      longitudinal_shift_minus_dt_conformal_metric, 0.);
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    detail::fully_contract(make_not_null(&longitudinal_shift_square),
                           longitudinal_shift_minus_dt_conformal_metric,
                           longitudinal_shift_minus_dt_conformal_metric);
  } else {
    detail::fully_contract(make_not_null(&longitudinal_shift_square),
                           longitudinal_shift_minus_dt_conformal_metric,
                           longitudinal_shift_minus_dt_conformal_metric,
                           conformal_metric->get());
  }
  // Add shift terms to Hamiltonian and lapse equations
  get(*hamiltonian_constraint) -= 0.03125 * pow<7>(get(conformal_factor)) /
                                  square(get(lapse_times_conformal_factor)) *
                                  get(longitudinal_shift_square);
  get(*lapse_equation) += pow<6>(get(conformal_factor)) /
                          get(lapse_times_conformal_factor) *
                          get(longitudinal_shift_square) * 7. / 32.;
  // Compute shift source
  // Begin with extrinsic curvature term
  auto extrinsic_curvature_trace_gradient_term =
      make_with_value<tnsr::I<DataVector, 3>>(
          extrinsic_curvature_trace_gradient, 0.);
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    get<0>(extrinsic_curvature_trace_gradient_term) =
        get<0>(extrinsic_curvature_trace_gradient);
    get<1>(extrinsic_curvature_trace_gradient_term) = get<1>(extrinsic_curvature_trace_gradient);
    get<2>(extrinsic_curvature_trace_gradient_term) =
        get<2>(extrinsic_curvature_trace_gradient);
  } else {
    raise_or_lower_index(
        make_not_null(&extrinsic_curvature_trace_gradient_term),
        extrinsic_curvature_trace_gradient, inv_conformal_metric->get());
  }
  for (size_t i = 0; i < 3;++i) {
    extrinsic_curvature_trace_gradient_term.get(i) *=
        4. / 3. * get(lapse_times_conformal_factor) / get(conformal_factor);
    momentum_constraint->get(i) +=
        extrinsic_curvature_trace_gradient_term.get(i);
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
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    get<0>(lapse_deriv_term_lo) = get<0>(lapse_deriv_term);
    get<1>(lapse_deriv_term_lo) = get<1>(lapse_deriv_term);
    get<2>(lapse_deriv_term_lo) = get<2>(lapse_deriv_term);
  } else {
    raise_or_lower_index(make_not_null(&lapse_deriv_term_lo), lapse_deriv_term,
                         conformal_metric->get());
  }
  for (size_t i = 0; i < 3; ++i) {
    // Add momentum density term
    momentum_constraint->get(i) +=
        16. * M_PI * get(lapse_times_conformal_factor) *
        pow<3>(get(conformal_factor)) * momentum_density.get(i);
    // Add longitudinal shift term
    for (size_t j = 0; j < 3; ++j) {
      momentum_constraint->get(i) +=
          longitudinal_shift_minus_dt_conformal_metric.get(i, j) *
          lapse_deriv_term_lo.get(j);
    }
    momentum_constraint->get(i) -= minus_div_dt_conformal_metric.get(i);
  }
}

void add_curved_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
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
  add_momentum_sources_impl<Geometry::Curved>(
      hamiltonian_constraint, lapse_equation, momentum_constraint,
      momentum_density, extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, minus_div_dt_conformal_metric, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
}

void add_flat_cartesian_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  add_momentum_sources_impl<Geometry::FlatCartesian>(
      hamiltonian_constraint, lapse_equation, momentum_constraint,
      momentum_density, extrinsic_curvature_trace_gradient, std::nullopt,
      std::nullopt, minus_div_dt_conformal_metric, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
}

template <Geometry ConformalGeometry>
void add_linearized_momentum_sources_impl(
    const gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const std::optional<std::reference_wrapper<const tnsr::ii<DataVector, 3>>>
        conformal_metric,
    const std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
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
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    ASSERT(not conformal_metric.has_value() and
               not inv_conformal_metric.has_value(),
           "You don't need to pass a conformal metric to this function when it "
           "is specialized for a flat conformal geometry in Cartesian "
           "coordinates.");
  } else {
    ASSERT(conformal_metric.has_value() and inv_conformal_metric.has_value(),
           "You must pass a conformal metric to this function when it is "
           "specialized for a curved conformal geometry.");
  }
  auto longitudinal_shift_square = make_with_value<Scalar<DataVector>>(
      longitudinal_shift_minus_dt_conformal_metric, 0.);
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    detail::fully_contract(make_not_null(&longitudinal_shift_square),
                           longitudinal_shift_minus_dt_conformal_metric,
                           longitudinal_shift_minus_dt_conformal_metric);
  } else {
    detail::fully_contract(make_not_null(&longitudinal_shift_square),
                           longitudinal_shift_minus_dt_conformal_metric,
                           longitudinal_shift_minus_dt_conformal_metric,
                           conformal_metric->get());
  }
  auto longitudinal_shift_dot_correction = make_with_value<Scalar<DataVector>>(
      longitudinal_shift_minus_dt_conformal_metric, 0.);
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    detail::fully_contract(make_not_null(&longitudinal_shift_dot_correction),
                           longitudinal_shift_minus_dt_conformal_metric,
                           longitudinal_shift_correction);
  } else {
    detail::fully_contract(make_not_null(&longitudinal_shift_dot_correction),
                           longitudinal_shift_minus_dt_conformal_metric,
                           longitudinal_shift_correction,
                           conformal_metric->get());
  }
  // Add shift terms to Hamiltonian and lapse equations
  get(*linearized_hamiltonian_constraint) +=
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
  get(*linearized_lapse_equation) +=
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
  auto extrinsic_curvature_trace_gradient_term =
      make_with_value<tnsr::I<DataVector, 3>>(
          extrinsic_curvature_trace_gradient, 0.);
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    get<0>(extrinsic_curvature_trace_gradient_term) =
        get<0>(extrinsic_curvature_trace_gradient);
    get<1>(extrinsic_curvature_trace_gradient_term) =
        get<1>(extrinsic_curvature_trace_gradient);
    get<2>(extrinsic_curvature_trace_gradient_term) =
        get<2>(extrinsic_curvature_trace_gradient);
  } else {
    raise_or_lower_index(
        make_not_null(&extrinsic_curvature_trace_gradient_term),
        extrinsic_curvature_trace_gradient, inv_conformal_metric->get());
  }
  for (size_t i = 0; i < 3;++i) {
    extrinsic_curvature_trace_gradient_term.get(i) *=
        4. / 3. *
        (get(lapse_times_conformal_factor_correction) / get(conformal_factor) -
         get(lapse_times_conformal_factor) / square(get(conformal_factor)) *
             get(conformal_factor_correction));
    linearized_momentum_constraint->get(i) +=
        extrinsic_curvature_trace_gradient_term.get(i);
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
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
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
                         conformal_metric->get());
    raise_or_lower_index(make_not_null(&lapse_deriv_correction_term_lo),
                         lapse_deriv_correction_term, conformal_metric->get());
  }
  for (size_t i = 0; i < 3; ++i) {
    // Add momentum density term
    linearized_momentum_constraint->get(i) +=
        16. * M_PI *
        (pow<3>(get(conformal_factor)) *
             get(lapse_times_conformal_factor_correction) +
         3. * square(get(conformal_factor)) *
             get(lapse_times_conformal_factor) *
             get(conformal_factor_correction)) *
        momentum_density.get(i);
    // Add longitudinal shift term
    for (size_t j = 0; j < 3; ++j) {
      linearized_momentum_constraint->get(i) +=
          longitudinal_shift_correction.get(i, j) * lapse_deriv_term_lo.get(j) +
          longitudinal_shift_minus_dt_conformal_metric.get(i, j) *
              lapse_deriv_correction_term_lo.get(j);
    }
  }
}

void add_curved_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
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
  add_linearized_momentum_sources_impl<Geometry::Curved>(
      linearized_hamiltonian_constraint, linearized_lapse_equation,
      linearized_momentum_constraint, momentum_density,
      extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, conformal_factor, lapse_times_conformal_factor,
      conformal_factor_flux, lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
}

void add_flat_cartesian_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
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
  add_linearized_momentum_sources_impl<Geometry::FlatCartesian>(
      linearized_hamiltonian_constraint, linearized_lapse_equation,
      linearized_momentum_constraint, momentum_density,
      extrinsic_curvature_trace_gradient, std::nullopt, std::nullopt,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DTYPE(_, data)                       \
  template void detail::fully_contract(                  \
      gsl::not_null<Scalar<DTYPE(data)>*> result,        \
      const tnsr::II<DTYPE(data), 3>& tensor1,           \
      const tnsr::II<DTYPE(data), 3>& tensor2) noexcept; \
  template void detail::fully_contract(                  \
      gsl::not_null<Scalar<DTYPE(data)>*> result,        \
      const tnsr::II<DTYPE(data), 3>& tensor1,           \
      const tnsr::II<DTYPE(data), 3>& tensor2,           \
      const tnsr::ii<DTYPE(data), 3>& metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE, (double, DataVector))

#undef INSTANTIATE_DTYPE
#undef GEOM
#undef DTYPE

}  // namespace Xcts
