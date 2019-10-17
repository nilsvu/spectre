// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Punctures {

void sources(const gsl::not_null<Scalar<DataVector>*> source_for_field,
             const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
             const Scalar<DataVector>& field) noexcept {
  get(*source_for_field) =
      -get(beta) / pow<7>(get(alpha) * (get(field) + 1.) + 1.);
}

void linearized_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field_correction,
    const Scalar<DataVector>& field, const Scalar<DataVector>& alpha,
    const Scalar<DataVector>& beta,
    const Scalar<DataVector>& field_correction) noexcept {
  get(*source_for_field_correction) =
      7. * get(alpha) * get(beta) /
      pow<8>(get(alpha) * (get(field) + 1.) + 1.) * get(field_correction);
}

}  // namespace Punctures

// Instantiate derivative templates
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/Systems/Punctures/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

template <size_t Dim>
using nonlinear_fluxes_tags_list =
    db::get_variables_tags_list<db::add_tag_prefix<
        ::Tags::Flux, typename Punctures::FirstOrderSystem::variables_tag,
        tmpl::size_t<Dim>, Frame::Inertial>>;
template <size_t Dim>
using linear_fluxes_tags_list = db::get_variables_tags_list<db::add_tag_prefix<
    ::Tags::Flux, typename Punctures::LinearizedFirstOrderSystem::variables_tag,
    tmpl::size_t<Dim>, Frame::Inertial>>;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DERIVS(_, data)                                           \
  template Variables<                                                         \
      db::wrap_tags_in<Tags::div, nonlinear_fluxes_tags_list<DIM(data)>>>     \
  divergence<nonlinear_fluxes_tags_list<DIM(data)>, DIM(data),                \
             Frame::Inertial>(                                                \
      const Variables<nonlinear_fluxes_tags_list<DIM(data)>>&,                \
      const Mesh<DIM(data)>&,                                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&) noexcept;                      \
  template Variables<                                                         \
      db::wrap_tags_in<Tags::div, linear_fluxes_tags_list<DIM(data)>>>        \
  divergence<linear_fluxes_tags_list<DIM(data)>, DIM(data), Frame::Inertial>( \
      const Variables<linear_fluxes_tags_list<DIM(data)>>&,                   \
      const Mesh<DIM(data)>&,                                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DERIVS, (1, 2, 3))

#undef INSTANTIATE_DERIVS
#undef DIM
