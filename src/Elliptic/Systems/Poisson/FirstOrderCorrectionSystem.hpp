// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename>
class Variables;
}  // namespace Tags

namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
/// \endcond

namespace Poisson {

template <size_t Dim, template <typename> class OperatorPrefix,
          typename FieldTag, typename AuxiliaryFieldTag>
struct ComputeFirstOrderCorrectionOperatorAction {
  using argument_tags =
      tmpl::list<Tags::deriv<FieldTag, tmpl::size_t<Dim>, Frame::Inertial>,
                 AuxiliaryFieldTag, Tags::Mesh<Dim>,
                 Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                       Tags::Coordinates<Dim, Frame::Logical>>>;
  using return_tags =
      db::wrap_tags_in<OperatorPrefix, tmpl::list<FieldTag, AuxiliaryFieldTag>>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr auto apply = ComputeFirstOrderOperatorAction<Dim>::apply;
};

template <size_t Dim, typename FieldTag, typename AuxiliaryFieldTag>
struct ComputeFirstOrderCorrectionNormalDotFluxes {
  using argument_tags =
      tmpl::list<FieldTag, AuxiliaryFieldTag,
                 Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>;
  using return_tags = db::wrap_tags_in<::Tags::NormalDotFlux,
                                       tmpl::list<FieldTag, AuxiliaryFieldTag>>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr auto apply = ComputeFirstOrderNormalDotFluxes<Dim>::apply;
};

template <size_t Dim, typename FieldTag, typename AuxiliaryFieldTag>
struct FirstOrderCorrectionInternalPenaltyFlux
    : public FirstOrderInternalPenaltyFlux<Dim> {
 public:
  using FirstOrderInternalPenaltyFlux<Dim>::FirstOrderInternalPenaltyFlux;

  using argument_tags =
      tmpl::list<FieldTag,
                 Tags::deriv<FieldTag, tmpl::size_t<Dim>, Frame::Inertial>,
                 Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>;
  using const_global_cache_tags = tmpl::list<>;

  using package_tags =
      typename FirstOrderInternalPenaltyFlux<Dim>::package_tags;
};

template <size_t Dim>
struct FirstOrderCorrectionSystem {
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using nonlinear_fields_tag =
      Tags::Variables<tmpl::list<Field, AuxiliaryField<Dim>>>;
  using fields_tag = db::add_tag_prefix<NonlinearSolver::Tags::Correction,
                                        nonlinear_fields_tag>;
  using impose_boundary_conditions_on_fields = tmpl::list<Field>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;

  // The bulk contribution to the linear operator action
  using compute_nonlinear_operator_action =
      ComputeFirstOrderCorrectionOperatorAction<
          Dim, NonlinearSolver::Tags::OperatorAppliedTo, Field,
          AuxiliaryField<Dim>>;
  using compute_operator_action = ComputeFirstOrderCorrectionOperatorAction<
      Dim, LinearSolver::Tags::OperatorAppliedTo,
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<Field>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<AuxiliaryField<Dim>>>>;

  // The interface normal dotted into the fluxes that is required by the strong
  // flux lifting operation
  using nonlinear_normal_dot_fluxes =
      ComputeFirstOrderCorrectionNormalDotFluxes<Dim, Field,
                                                 AuxiliaryField<Dim>>;
  using normal_dot_fluxes = ComputeFirstOrderCorrectionNormalDotFluxes<
      Dim,
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<Field>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<AuxiliaryField<Dim>>>>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;

  // The tags to instantiate derivative functions for
  using gradient_tags = tmpl::list<
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<Field>>>;
  using nonlinear_gradient_tags = tmpl::list<Field>;
  using divergence_tags = tmpl::list<AuxiliaryField<Dim>>;
};

}  // namespace Poisson
