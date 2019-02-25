// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tensor

namespace Xcts {

template <size_t Dim>
void ComputeFirstOrderNonlinearOperatorAction<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_divergence_operator,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        hamiltonian_auxiliary_operator,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, Dim>& conformal_factor_gradient,
    const Scalar<DataVector>& energy_density, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inverse_jacobian) noexcept {
  // Compute gradient manually so that we don't have to put it in the DataBox
  auto grad_vars =
      make_with_value<Variables<tmpl::list<Tags::ConformalFactor<DataVector>>>>(
          conformal_factor, 0.);
  get<Tags::ConformalFactor<DataVector>>(grad_vars) = conformal_factor;
  auto computed_conformal_factor_gradient =
      get<::Tags::deriv<Tags::ConformalFactor<DataVector>, tmpl::size_t<Dim>,
                        Frame::Inertial>>(
          partial_derivatives<tmpl::list<Tags::ConformalFactor<DataVector>>>(
              grad_vars, mesh, inverse_jacobian));

  Poisson::ComputeFirstOrderOperatorAction<Dim>::apply(
      hamiltonian_divergence_operator, hamiltonian_auxiliary_operator,
      std::move(computed_conformal_factor_gradient), conformal_factor_gradient,
      mesh, inverse_jacobian);

  get(*hamiltonian_divergence_operator) -=
      2. * M_PI * get(energy_density) * pow<5>(get(conformal_factor));
}

template <size_t Dim>
void ComputeFirstOrderLinearOperatorAction<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_divergence_operator,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        hamiltonian_auxiliary_operator,
    const Scalar<DataVector>& conformal_factor_correction,
    const tnsr::i<DataVector, Dim>&
        computed_conformal_factor_gradient_correction,
    const tnsr::I<DataVector, Dim>&
        auxiliary_conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& energy_density, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inverse_jacobian) noexcept {
  Poisson::ComputeFirstOrderOperatorAction<Dim>::apply(
      hamiltonian_divergence_operator, hamiltonian_auxiliary_operator,
      computed_conformal_factor_gradient_correction,
      auxiliary_conformal_factor_gradient_correction, mesh, inverse_jacobian);

  get(*hamiltonian_divergence_operator) -= 10. * M_PI * get(energy_density) *
                                           pow<4>(get(conformal_factor)) *
                                           get(conformal_factor_correction);
}

template <size_t Dim, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
void ComputeFirstOrderNormalDotFluxes<Dim, ConformalFactorTag,
                                      ConformalFactorGradientTag>::
    apply(const gsl::not_null<Scalar<DataVector>*> hamiltonian_normal_dot_flux,
          const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
              hamiltonian_auxiliary_normal_dot_flux,
          const Scalar<DataVector>& conformal_factor,
          const tnsr::I<DataVector, Dim, Frame::Inertial>&
              conformal_factor_gradient,
          const tnsr::i<DataVector, Dim, Frame::Inertial>&
              interface_unit_normal) noexcept {
  Poisson::ComputeFirstOrderNormalDotFluxes<Dim>::apply(
      hamiltonian_normal_dot_flux, hamiltonian_auxiliary_normal_dot_flux,
      conformal_factor, conformal_factor_gradient, interface_unit_normal);
}

template <size_t Dim, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
void FirstOrderInternalPenaltyFlux<Dim, ConformalFactorTag,
                                   ConformalFactorGradientTag>::
    package_data(const gsl::not_null<Variables<package_tags>*> packaged_data,
                 const Scalar<DataVector>& conformal_factor,
                 const tnsr::i<DataVector, Dim, Frame::Inertial>&
                     conformal_factor_gradient,
                 const tnsr::i<DataVector, Dim, Frame::Inertial>&
                     interface_unit_normal) const noexcept {
  // Can we also forward this to Poisson::FirstOrderInternalPenaltyFlux?
  get<ConformalFactorTag>(*packaged_data) = conformal_factor;
  for (size_t d = 0; d < Dim; d++) {
    get<NormalTimesConformalFactorFlux>(*packaged_data).get(d) =
        interface_unit_normal.get(d) * get(conformal_factor);
  }
  get<NormalDotConformalFactorGradientFlux>(*packaged_data).get() =
      get<0>(interface_unit_normal) * get<0>(conformal_factor_gradient);
  for (size_t d = 1; d < Dim; d++) {
    get<NormalDotConformalFactorGradientFlux>(*packaged_data).get() +=
        interface_unit_normal.get(d) * conformal_factor_gradient.get(d);
  }
}

template <size_t Dim, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
void FirstOrderInternalPenaltyFlux<Dim, ConformalFactorTag,
                                   ConformalFactorGradientTag>::
operator()(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_numerical_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        hamiltonian_auxiliary_numerical_flux,
    const Scalar<DataVector>& conformal_factor_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_times_conformal_factor_interior,
    const Scalar<DataVector>& normal_dot_conformal_factor_gradient_interior,
    const Scalar<DataVector>& conformal_factor_exterior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        minus_normal_times_conformal_factor_exterior,
    const Scalar<DataVector>&
        minus_normal_dot_conformal_factor_gradient_exterior) const noexcept {
  poisson_flux_(
      hamiltonian_numerical_flux, hamiltonian_auxiliary_numerical_flux,
      conformal_factor_interior, normal_times_conformal_factor_interior,
      normal_dot_conformal_factor_gradient_interior, conformal_factor_exterior,
      minus_normal_times_conformal_factor_exterior,
      minus_normal_dot_conformal_factor_gradient_exterior);
}

template <size_t Dim, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
void FirstOrderInternalPenaltyFlux<Dim, ConformalFactorTag,
                                   ConformalFactorGradientTag>::
    compute_dirichlet_boundary(
        const gsl::not_null<Scalar<DataVector>*> hamiltonian_numerical_flux,
        const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
            hamiltonian_auxiliary_numerical_flux,
        const Scalar<DataVector>& dirichlet_conformal_factor,
        const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
        const noexcept {
  poisson_flux_.compute_dirichlet_boundary(
      hamiltonian_numerical_flux, hamiltonian_auxiliary_numerical_flux,
      dirichlet_conformal_factor, interface_unit_normal);
}

}  // namespace Xcts

// Instantiate needed gradient and divergence templates
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using variables_tags =
    typename Xcts::FirstOrderSystem<Dim>::variables_tag::type::tags_list;
template <size_t Dim>
using grad_tags = typename Xcts::FirstOrderSystem<Dim>::gradient_tags;
template <size_t Dim>
using div_tags = typename Xcts::FirstOrderSystem<Dim>::divergence_tags;

template <size_t Dim>
using extra_grad_tags = tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>;
template <size_t Dim>
using nonlin_variables_tags =
    typename Xcts::FirstOrderSystem<Dim>::nonlinear_fields_tag::type::tags_list;
template <size_t Dim>
using nonlin_grad_tags =
    typename Xcts::FirstOrderSystem<Dim>::nonlinear_gradient_tags;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template class Xcts::ComputeFirstOrderLinearOperatorAction<DIM(data)>;      \
  template class Xcts::ComputeFirstOrderNonlinearOperatorAction<DIM(data)>;   \
  template class Xcts::ComputeFirstOrderNormalDotFluxes<                      \
      DIM(data), Xcts::Tags::ConformalFactor<DataVector>,                     \
      Xcts::Tags::ConformalFactorGradient<DIM(data), Frame::Inertial,         \
                                          DataVector>>;                       \
  template class Xcts::ComputeFirstOrderNormalDotFluxes<                      \
      DIM(data),                                                              \
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<          \
          Xcts::Tags::ConformalFactor<DataVector>>>,                          \
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<          \
          Xcts::Tags::ConformalFactorGradient<DIM(data), Frame::Inertial,     \
                                              DataVector>>>>;                 \
  template class Xcts::FirstOrderInternalPenaltyFlux<                         \
      DIM(data), Xcts::Tags::ConformalFactor<DataVector>,                     \
      Xcts::Tags::ConformalFactorGradient<DIM(data), Frame::Inertial,         \
                                          DataVector>>;                       \
  template class Xcts::FirstOrderInternalPenaltyFlux<                         \
      DIM(data),                                                              \
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<          \
          Xcts::Tags::ConformalFactor<DataVector>>>,                          \
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<          \
          Xcts::Tags::ConformalFactorGradient<DIM(data), Frame::Inertial,     \
                                              DataVector>>>>;                 \
  template Variables<                                                         \
      db::wrap_tags_in<Tags::deriv, grad_tags<DIM(data)>,                     \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>             \
  partial_derivatives<grad_tags<DIM(data)>, variables_tags<DIM(data)>,        \
                      DIM(data), Frame::Inertial>(                            \
      const Variables<variables_tags<DIM(data)>>&, const Mesh<DIM(data)>&,    \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&) noexcept;                      \
  template Variables<                                                         \
      db::wrap_tags_in<Tags::deriv, nonlin_grad_tags<DIM(data)>,              \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>             \
  partial_derivatives<nonlin_grad_tags<DIM(data)>,                            \
                      nonlin_variables_tags<DIM(data)>, DIM(data),            \
                      Frame::Inertial>(                                       \
      const Variables<nonlin_variables_tags<DIM(data)>>&,                     \
      const Mesh<DIM(data)>&,                                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&) noexcept;                      \
  template Variables<                                                         \
      db::wrap_tags_in<Tags::deriv, extra_grad_tags<DIM(data)>,               \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>             \
  partial_derivatives<extra_grad_tags<DIM(data)>, extra_grad_tags<DIM(data)>, \
                      DIM(data), Frame::Inertial>(                            \
      const Variables<extra_grad_tags<DIM(data)>>&, const Mesh<DIM(data)>&,   \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
