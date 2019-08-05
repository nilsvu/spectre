// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
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
void first_order_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& conformal_factor_gradient,
    const Scalar<DataVector>& energy_density) noexcept {
  Poisson::first_order_sources(source_for_conformal_factor,
                               source_for_conformal_factor_gradient,
                               conformal_factor, conformal_factor_gradient);
  get(*source_for_conformal_factor) -=
      2. * M_PI * get(energy_density) * pow<5>(get(conformal_factor));
}

template <size_t Dim>
void first_order_linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& energy_density) noexcept {
  Poisson::first_order_sources(source_for_conformal_factor_correction,
                               source_for_conformal_factor_gradient_correction,
                               conformal_factor_correction,
                               conformal_factor_gradient_correction);
  get(*source_for_conformal_factor_correction) -=
      10. * M_PI * get(energy_density) * pow<4>(get(conformal_factor)) *
      get(conformal_factor_correction);
}

template <size_t Dim>
void first_order_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_lapse_times_conformal_factor_gradient,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace) noexcept {
  Poisson::first_order_sources(source_for_lapse_times_conformal_factor,
                               source_for_lapse_times_conformal_factor_gradient,
                               lapse_times_conformal_factor,
                               lapse_times_conformal_factor_gradient);
  get(*source_for_lapse_times_conformal_factor) +=
      2. * M_PI * (get(energy_density) + 2. * get(stress_trace)) *
      get(lapse_times_conformal_factor) * pow<4>(get(conformal_factor));
}

template <size_t Dim>
void first_order_linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_lapse_times_conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace) noexcept {
  Poisson::first_order_sources(
      source_for_lapse_times_conformal_factor_correction,
      source_for_lapse_times_conformal_factor_gradient_correction,
      lapse_times_conformal_factor_correction,
      lapse_times_conformal_factor_gradient_correction);
  get(*source_for_lapse_times_conformal_factor_correction) +=
      2. * M_PI * (get(energy_density) + 2 * get(stress_trace)) *
      (pow<4>(get(conformal_factor)) *
           get(lapse_times_conformal_factor_correction) +
       4. * get(lapse_times_conformal_factor) * pow<3>(get(conformal_factor)) *
           get(conformal_factor_correction));
}

}  // namespace Xcts

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_EQUATIONS(_, data)                                       \
  template void Xcts::first_order_hamiltonian_sources<DIM(data)>(            \
      const gsl::not_null<Scalar<DataVector>*>,                              \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>, \
      const Scalar<DataVector>&,                                             \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&) noexcept;                                   \
  template void Xcts::first_order_linearized_hamiltonian_sources<DIM(data)>( \
      const gsl::not_null<Scalar<DataVector>*>,                              \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>, \
      const Scalar<DataVector>&,                                             \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&, const Scalar<DataVector>&) noexcept;        \
  template void Xcts::first_order_lapse_sources<DIM(data)>(                  \
      const gsl::not_null<Scalar<DataVector>*>,                              \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>, \
      const Scalar<DataVector>&,                                             \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                  \
      const Scalar<DataVector>&) noexcept;                                   \
  template void Xcts::first_order_linearized_lapse_sources<DIM(data)>(       \
      const gsl::not_null<Scalar<DataVector>*>,                              \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>, \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                  \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                  \
      const Scalar<DataVector>&, const Scalar<DataVector>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_EQUATIONS, (1, 2, 3))

#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep

#define SYSTEM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define LINEAR_VARS_TAGS_LIST(data)                                          \
  db::get_variables_tags_list<                                               \
      db::add_tag_prefix<                                                    \
          ::Tags::Flux,                                                      \
          db::add_tag_prefix<LinearSolver::Tags::Operand,                    \
                             typename SYSTEM(data) <                         \
                                 DIM(data)>::linearized_system::fields_tag>, \
      tmpl::size_t<DIM(data)>, Frame::Inertial>>

#define NONLINEAR_VARS_TAGS_LIST(data)                                   \
  db::get_variables_tags_list<                                           \
      db::add_tag_prefix<::Tags::Flux,                                   \
                         typename SYSTEM(data) < DIM(data)>::fields_tag, \
      tmpl::size_t<DIM(data)>, Frame::Inertial>>

#define INSTANTIATE_DERIVATIVES(_, data)                                       \
  template Variables<db::wrap_tags_in<Tags::div, LINEAR_VARS_TAGS_LIST(data)>> \
  divergence<LINEAR_VARS_TAGS_LIST(data), DIM(data), Frame::Inertial>(         \
      const Variables<LINEAR_VARS_TAGS_LIST(data)>&, const Mesh<DIM(data)>&,   \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,             \
                            Frame::Inertial>&) noexcept;                       \
  template Variables<                                                          \
      db::wrap_tags_in<Tags::div, NONLINEAR_VARS_TAGS_LIST(data)>>             \
  divergence<NONLINEAR_VARS_TAGS_LIST(data), DIM(data), Frame::Inertial>(      \
      const Variables<NONLINEAR_VARS_TAGS_LIST(data)>&,                        \
      const Mesh<DIM(data)>&,                                                  \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,             \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DERIVATIVES, (1, 2, 3),
                        (Xcts::FirstOrderHamiltonianSystem,
                         Xcts::FirstOrderHamiltonianAndLapseSystem))

#undef INSTANTIATE_EQUATIONS
#undef INSTANTIATE_DERIVATIVES
#undef DIM
#undef SYSTEM
