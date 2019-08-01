// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/FirstOrderCorrectionSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

// Instantiate derivative templates
template <size_t Dim>
using nonlinear_variables_tag =
    typename Poisson::FirstOrderCorrectionSystem<Dim>::variables_tag;
template <size_t Dim>
using nonlinear_fluxes_tags_list = db::get_variables_tags_list<
    db::add_tag_prefix<::Tags::Flux, nonlinear_variables_tag<Dim>,
                       tmpl::size_t<Dim>, Frame::Inertial>>;
template <size_t Dim>
using linear_variables_tag =
    typename Poisson::LinearizedFirstOrderCorrectionSystem<Dim>::variables_tag;
template <size_t Dim>
using linear_fluxes_tags_list = db::get_variables_tags_list<
    db::add_tag_prefix<::Tags::Flux, linear_variables_tag<Dim>,
                       tmpl::size_t<Dim>, Frame::Inertial>>;

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
