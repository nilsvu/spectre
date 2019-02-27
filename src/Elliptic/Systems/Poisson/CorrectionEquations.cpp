// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/FirstOrderCorrectionSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tensor

// Instantiate needed gradient and divergence templates
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using variables_tags = typename Poisson::FirstOrderCorrectionSystem<
    Dim>::variables_tag::type::tags_list;
template <size_t Dim>
using grad_tags =
    typename Poisson::FirstOrderCorrectionSystem<Dim>::gradient_tags;
template <size_t Dim>
using div_tags =
    typename Poisson::FirstOrderCorrectionSystem<Dim>::divergence_tags;
template <size_t Dim>
using nonlin_variables_tags = typename Poisson::FirstOrderCorrectionSystem<
    Dim>::nonlinear_fields_tag::type::tags_list;
template <size_t Dim>
using nonlin_grad_tags =
    typename Poisson::FirstOrderCorrectionSystem<Dim>::nonlinear_gradient_tags;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                             \
  template class Poisson::ComputeFirstOrderCorrectionOperatorAction<       \
      DIM(data), NonlinearSolver::Tags::OperatorAppliedTo, Poisson::Field, \
      Poisson::AuxiliaryField<DIM(data)>>;                                 \
  template class Poisson::ComputeFirstOrderCorrectionOperatorAction<       \
      DIM(data), LinearSolver::Tags::OperatorAppliedTo,                    \
      LinearSolver::Tags::Operand<                                         \
          NonlinearSolver::Tags::Correction<Poisson::Field>>,              \
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<       \
          Poisson::AuxiliaryField<DIM(data)>>>>;                           \
  template class Poisson::ComputeFirstOrderCorrectionNormalDotFluxes<      \
      DIM(data), Poisson::Field, Poisson::AuxiliaryField<DIM(data)>>;      \
  template class Poisson::ComputeFirstOrderCorrectionNormalDotFluxes<      \
      DIM(data),                                                           \
      LinearSolver::Tags::Operand<                                         \
          NonlinearSolver::Tags::Correction<Poisson::Field>>,              \
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<       \
          Poisson::AuxiliaryField<DIM(data)>>>>;                           \
  template class Poisson::FirstOrderCorrectionInternalPenaltyFlux<         \
      DIM(data), Poisson::Field, Poisson::AuxiliaryField<DIM(data)>>;      \
  template class Poisson::FirstOrderCorrectionInternalPenaltyFlux<         \
      DIM(data),                                                           \
      LinearSolver::Tags::Operand<                                         \
          NonlinearSolver::Tags::Correction<Poisson::Field>>,              \
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<       \
          Poisson::AuxiliaryField<DIM(data)>>>>;                           \
  template Variables<                                                      \
      db::wrap_tags_in<Tags::deriv, grad_tags<DIM(data)>,                  \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>          \
  partial_derivatives<grad_tags<DIM(data)>, variables_tags<DIM(data)>,     \
                      DIM(data), Frame::Inertial>(                         \
      const Variables<variables_tags<DIM(data)>>&, const Mesh<DIM(data)>&, \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,         \
                            Frame::Inertial>&) noexcept;                   \
  template Variables<                                                      \
      db::wrap_tags_in<Tags::deriv, nonlin_grad_tags<DIM(data)>,           \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>          \
  partial_derivatives<nonlin_grad_tags<DIM(data)>,                         \
                      nonlin_variables_tags<DIM(data)>, DIM(data),         \
                      Frame::Inertial>(                                    \
      const Variables<nonlin_variables_tags<DIM(data)>>&,                  \
      const Mesh<DIM(data)>&,                                              \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,         \
                            Frame::Inertial>&) noexcept;                   \
  template Variables<db::wrap_tags_in<Tags::div, div_tags<DIM(data)>>>     \
  divergence<div_tags<DIM(data)>, DIM(data), Frame::Inertial>(             \
      const Variables<div_tags<DIM(data)>>&, const Mesh<DIM(data)>&,       \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,         \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
