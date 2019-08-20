// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson {

template <size_t Dim>
void flux(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_for_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_gradient) noexcept {
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <size_t Dim>
void first_order_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_for_field,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_for_auxiliary_field,
    const Scalar<DataVector>& field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_field) noexcept {
  *flux_for_field = auxiliary_field;
  *flux_for_auxiliary_field =
      make_with_value<tnsr::IJ<DataVector, Dim, Frame::Inertial>>(field, 0.);
  for (size_t d = 0; d < Dim; d++) {
    flux_for_auxiliary_field->get(d, d) = get(field);
  }
}

template <size_t Dim>
void first_order_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_auxiliary_field,
    const Scalar<DataVector>& field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_field) noexcept {
  *source_for_field = make_with_value<Scalar<DataVector>>(field, 0.);
  *source_for_auxiliary_field = auxiliary_field;
}

}  // namespace Poisson

// Instantiate needed derivative templates
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"

template <size_t Dim>
using variables_tag =
    db::add_tag_prefix<LinearSolver::Tags::Operand,
                       typename Poisson::SecondOrderSystem<Dim>::fields_tag>;
template <size_t Dim>
using gradients_tags_list = db::get_variables_tags_list<variables_tag<Dim>>;
template <size_t Dim>
using fluxes_tags_list = db::get_variables_tags_list<db::add_tag_prefix<
    ::Tags::Flux, variables_tag<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                 \
  template void Poisson::flux<DIM(data)>(                                      \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,   \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;        \
  template void Poisson::first_order_fluxes<DIM(data)>(                        \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,   \
      const gsl::not_null<tnsr::IJ<DataVector, DIM(data), Frame::Inertial>*>,  \
      const Scalar<DataVector>&,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&) noexcept;        \
  template void Poisson::first_order_sources<DIM(data)>(                       \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,   \
      const Scalar<DataVector>&,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&) noexcept;        \
  template Variables<                                                          \
      db::wrap_tags_in<::Tags::deriv, gradients_tags_list<DIM(data)>,          \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>              \
  partial_derivatives<gradients_tags_list<DIM(data)>>(                         \
      const Variables<gradients_tags_list<DIM(data)>>&,                        \
      const Mesh<DIM(data)>&,                                                  \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,             \
                            Frame::Inertial>&) noexcept;                       \
  template Variables<db::wrap_tags_in<Tags::div, fluxes_tags_list<DIM(data)>>> \
  divergence(const Variables<fluxes_tags_list<DIM(data)>>&,                    \
             const Mesh<DIM(data)>&,                                           \
             const InverseJacobian<DataVector, DIM(data), Frame::Logical,      \
                                   Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
