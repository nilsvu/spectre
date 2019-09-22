// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {

template <size_t Dim>
void longitudinal_shift(
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_for_shift,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain) noexcept {
  auto shift_strain_trace_term = get<0, 0>(shift_strain);
  for (size_t d = 1; d < Dim; d++) {
    shift_strain_trace_term += shift_strain.get(d, d);
  }
  shift_strain_trace_term /= 3.;
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      flux_for_shift->get(i, j) = shift_strain.get(i, j);
    }
    flux_for_shift->get(i, i) -= shift_strain_trace_term;
  }
}

template <size_t Dim>
void momentum_auxiliary_fluxes(
    const gsl::not_null<tnsr::Ijj<DataVector, Dim, Frame::Inertial>*>
        flux_for_shift_strain,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      flux_for_shift_strain->get(i, i, j) = 0.5 * (shift.get(i) + shift.get(j));
    }
  }
}

void hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& conformal_factor) noexcept {
  get(*source_for_conformal_factor) =
      -2. * M_PI * get(energy_density) * pow<5>(get(conformal_factor));
}

void linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  get(*source_for_conformal_factor_correction) =
      -10. * M_PI * get(energy_density) * pow<4>(get(conformal_factor)) *
      get(conformal_factor_correction);
}

void lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  get(*source_for_lapse_times_conformal_factor) =
      2. * M_PI * (get(energy_density) + 2. * get(stress_trace)) *
      get(lapse_times_conformal_factor) * pow<4>(get(conformal_factor));
}

void linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  get(*source_for_lapse_times_conformal_factor_correction) =
      2. * M_PI * (get(energy_density) + 2 * get(stress_trace)) *
      (pow<4>(get(conformal_factor)) *
           get(lapse_times_conformal_factor_correction) +
       4. * get(lapse_times_conformal_factor) * pow<3>(get(conformal_factor)) *
           get(conformal_factor_correction));
}

template <size_t Dim>
void momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_shift,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain) noexcept {
  auto longitudinal_shift =
      make_with_value<tnsr::IJ<DataVector, Dim, Frame::Inertial>>(shift_strain,
                                                                  0.);
  Xcts::longitudinal_shift(make_not_null(&longitudinal_shift), shift_strain);
  // Add shift terms to Hamiltonian and lapse equations
  const auto longitudinal_shift_square =
      pointwise_l2_norm_square(longitudinal_shift);
  get(*source_for_conformal_factor) -=
      pow<7>(get(conformal_factor)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_square) / 32.;
  get(*source_for_lapse_times_conformal_factor) +=
      pow<6>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
      get(longitudinal_shift_square) * 7. / 32.;
  // Compute shift source
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      source_for_shift->get(i) +=
          longitudinal_shift.get(i, j) *
          (lapse_times_conformal_factor_gradient.get(j) /
               get(lapse_times_conformal_factor) -
           7. * conformal_factor_gradient.get(j) / get(conformal_factor));
    }
    source_for_shift->get(i) += 16. * M_PI * get(lapse_times_conformal_factor) *
                                pow<3>(get(conformal_factor)) *
                                momentum_density.get(i);
  }
}

template <size_t Dim>
void linearized_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_shift_correction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        conformal_factor_gradient_correction,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient_correction,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
        shift_strain_correction) noexcept {
  auto longitudinal_shift =
      make_with_value<tnsr::IJ<DataVector, Dim, Frame::Inertial>>(shift_strain,
                                                                  0.);
  Xcts::longitudinal_shift(make_not_null(&longitudinal_shift), shift_strain);
  auto longitudinal_shift_correction =
      make_with_value<tnsr::IJ<DataVector, Dim, Frame::Inertial>>(
          shift_strain_correction, 0.);
  Xcts::longitudinal_shift(make_not_null(&longitudinal_shift_correction),
                           shift_strain_correction);
  // Add shift terms to Hamiltonian and lapse equations
  const auto longitudinal_shift_square =
      pointwise_l2_norm_square(longitudinal_shift);
  DataVector longitudinal_shift_dot_correction{conformal_factor.begin()->size(),
                                               0.};
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      longitudinal_shift_dot_correction +=
          longitudinal_shift.get(i, j) *
          longitudinal_shift_correction.get(i, j);
    }
  }
  get(*source_for_conformal_factor_correction) +=
      -7. / 32. * pow<6>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          get(longitudinal_shift_square) * get(conformal_factor_correction) +
      1. / 16. * pow<7>(get(conformal_factor)) /
          pow<3>(get(lapse_times_conformal_factor)) *
          get(longitudinal_shift_square) *
          get(lapse_times_conformal_factor_correction) -
      1. / 16. * pow<7>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          longitudinal_shift_dot_correction;
  get(*source_for_lapse_times_conformal_factor_correction) +=
      21. / 16. * pow<5>(get(conformal_factor)) /
          get(lapse_times_conformal_factor) * get(longitudinal_shift_square) *
          get(conformal_factor_correction) -
      7. / 32. * pow<6>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          get(longitudinal_shift_square) *
          get(lapse_times_conformal_factor_correction) +
      7. / 32. * pow<6>(get(conformal_factor)) /
          get(lapse_times_conformal_factor) * longitudinal_shift_dot_correction;
  // Compute shift source
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
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
        momentum_density.get(i);
  }
}

}  // namespace Xcts

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_EQUATIONS(_, data)                                         \
  template void Xcts::longitudinal_shift<DIM(data)>(                           \
      const gsl::not_null<tnsr::IJ<DataVector, DIM(data), Frame::Inertial>*>   \
          flux_for_shift,                                                      \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&                  \
          shift_strain) noexcept;                                              \
  template void Xcts::momentum_auxiliary_fluxes<DIM(data)>(                    \
      const gsl::not_null<tnsr::Ijj<DataVector, DIM(data), Frame::Inertial>*>  \
          flux_for_shift_strain,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift) noexcept;  \
  template void Xcts::momentum_sources<DIM(data)>(                             \
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,    \
      const gsl::not_null<Scalar<DataVector>*>                                 \
          source_for_lapse_times_conformal_factor,                             \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>    \
          source_for_shift,                                                    \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& momentum_density, \
      const Scalar<DataVector>& conformal_factor,                              \
      const Scalar<DataVector>& lapse_times_conformal_factor,                  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          conformal_factor_gradient,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          lapse_times_conformal_factor_gradient,                               \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&                  \
          shift_strain) noexcept;                                              \
  template void Xcts::linearized_momentum_sources<DIM(data)>(                  \
      const gsl::not_null<Scalar<DataVector>*>                                 \
          source_for_conformal_factor_correction,                              \
      const gsl::not_null<Scalar<DataVector>*>                                 \
          source_for_lapse_times_conformal_factor_correction,                  \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>    \
          source_for_shift_correction,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& momentum_density, \
      const Scalar<DataVector>& conformal_factor,                              \
      const Scalar<DataVector>& lapse_times_conformal_factor,                  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          conformal_factor_gradient,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          lapse_times_conformal_factor_gradient,                               \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& shift_strain,    \
      const Scalar<DataVector>& conformal_factor_correction,                   \
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,       \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          conformal_factor_gradient_correction,                                \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          lapse_times_conformal_factor_gradient_correction,                    \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&                  \
          shift_strain_correction) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_EQUATIONS, (1, 2, 3))

#define EQNS(data) BOOST_PP_TUPLE_ELEM(1, data)

// Instantiate derivative templates
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"

template <size_t Dim, Xcts::Equations EnabledEquations>
using nonlinear_fluxes_tags_list =
    db::get_variables_tags_list<db::add_tag_prefix<
        ::Tags::Flux,
        typename Xcts::FirstOrderSystem<Dim, EnabledEquations>::variables_tag,
        tmpl::size_t<Dim>, Frame::Inertial>>;
template <size_t Dim, Xcts::Equations EnabledEquations>
using linear_fluxes_tags_list = db::get_variables_tags_list<
    db::add_tag_prefix<::Tags::Flux,
                       typename Xcts::LinearizedFirstOrderSystem<
                           Dim, EnabledEquations>::variables_tag,
                       tmpl::size_t<Dim>, Frame::Inertial>>;

#define INSTANTIATE_DERIVS(_, data)                                        \
  template Variables<db::wrap_tags_in<                                     \
      Tags::div, nonlinear_fluxes_tags_list<DIM(data), EQNS(data)>>>       \
  divergence<nonlinear_fluxes_tags_list<DIM(data), EQNS(data)>, DIM(data), \
             Frame::Inertial>(                                             \
      const Variables<nonlinear_fluxes_tags_list<DIM(data), EQNS(data)>>&, \
      const Mesh<DIM(data)>&,                                              \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,         \
                            Frame::Inertial>&) noexcept;                   \
  template Variables<db::wrap_tags_in<                                     \
      Tags::div, linear_fluxes_tags_list<DIM(data), EQNS(data)>>>          \
  divergence<linear_fluxes_tags_list<DIM(data), EQNS(data)>, DIM(data),    \
             Frame::Inertial>(                                             \
      const Variables<linear_fluxes_tags_list<DIM(data), EQNS(data)>>&,    \
      const Mesh<DIM(data)>&,                                              \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,         \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DERIVS, (1, 2, 3),
                        (Xcts::Equations::Hamiltonian,
                         Xcts::Equations::HamiltonianAndLapse,
                         Xcts::Equations::HamiltonianLapseAndShift))

#undef INSTANTIATE_EQUATIONS
#undef INSTANTIATE_DERIVS
#undef DIM
