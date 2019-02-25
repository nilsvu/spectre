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
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {

template <size_t Dim>
void first_order_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_for_conformal_factor,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_for_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        conformal_factor_gradient) noexcept {
  Poisson::first_order_fluxes(std::move(flux_for_conformal_factor),
                              std::move(flux_for_conformal_factor_gradient),
                              conformal_factor, conformal_factor_gradient);
}

template <size_t Dim>
void first_order_sources(
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
void first_order_linearized_sources(
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

}  // namespace Xcts

// Instantiate needed divergence templates
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep

template <size_t Dim>
using linear_variables_tag = db::add_tag_prefix<
    LinearSolver::Tags::Operand,
    typename Xcts::FirstOrderSystem<Dim>::linearized_system::fields_tag>;
template <size_t Dim>
using linear_fluxes_tags_list = db::get_variables_tags_list<
    db::add_tag_prefix<::Tags::Flux, linear_variables_tag<Dim>,
                       tmpl::size_t<Dim>, Frame::Inertial>>;

template <size_t Dim>
using nonlinear_variables_tag =
    typename Xcts::FirstOrderSystem<Dim>::fields_tag;
template <size_t Dim>
using nonlinear_fluxes_tags_list = db::get_variables_tags_list<
    db::add_tag_prefix<::Tags::Flux, nonlinear_variables_tag<Dim>,
                       tmpl::size_t<Dim>, Frame::Inertial>>;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template void Xcts::first_order_fluxes<DIM(data)>(                          \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,  \
      const gsl::not_null<tnsr::IJ<DataVector, DIM(data), Frame::Inertial>*>, \
      const Scalar<DataVector>&,                                              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template void Xcts::first_order_sources<DIM(data)>(                         \
      const gsl::not_null<Scalar<DataVector>*>,                               \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,  \
      const Scalar<DataVector>&,                                              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&) noexcept;                                    \
  template void Xcts::first_order_linearized_sources<DIM(data)>(              \
      const gsl::not_null<Scalar<DataVector>*>,                               \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,  \
      const Scalar<DataVector>&,                                              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&, const Scalar<DataVector>&) noexcept;         \
  template Variables<                                                         \
      db::wrap_tags_in<Tags::div, linear_fluxes_tags_list<DIM(data)>>>        \
  divergence<linear_fluxes_tags_list<DIM(data)>, DIM(data), Frame::Inertial>( \
      const Variables<linear_fluxes_tags_list<DIM(data)>>&,                   \
      const Mesh<DIM(data)>&,                                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&) noexcept;                      \
  template Variables<                                                         \
      db::wrap_tags_in<Tags::div, nonlinear_fluxes_tags_list<DIM(data)>>>     \
  divergence<nonlinear_fluxes_tags_list<DIM(data)>, DIM(data),                \
             Frame::Inertial>(                                                \
      const Variables<nonlinear_fluxes_tags_list<DIM(data)>>&,                \
      const Mesh<DIM(data)>&,                                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
