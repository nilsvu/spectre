// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename>
class Variables;
}  // namespace Tags
/// \endcond

namespace Xcts {

template <size_t Dim>
struct FirstOrderSystem {
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using nonlinear_fields_tag = ::Tags::Variables<tmpl::list<
      Tags::ConformalFactor<DataVector>,
      Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>>>;
  using fields_tag = db::add_tag_prefix<NonlinearSolver::Tags::Correction,
                                        nonlinear_fields_tag>;
  using impose_boundary_conditions_on_fields =
      tmpl::list<Tags::ConformalFactor<DataVector>>;
  using background_tags = tmpl::list<gr::Tags::EnergyDensity<DataVector>>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;

  // The bulk contribution to the linear operator action
  using compute_nonlinear_operator_action =
      ComputeFirstOrderNonlinearOperatorAction<Dim>;
  using compute_operator_action = ComputeFirstOrderLinearOperatorAction<Dim>;

  // The interface normal dotted into the fluxes that is required by the strong
  // flux lifting operation
  using normal_dot_fluxes = ComputeFirstOrderNormalDotFluxes<
      Dim,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<Tags::ConformalFactor<DataVector>>>,
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
          Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>>>>;
  using nonlinear_normal_dot_fluxes = ComputeFirstOrderNormalDotFluxes<
      Dim, Tags::ConformalFactor<DataVector>,
      Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;

  // The tags to instantiate derivative functions for
  using gradient_tags = tmpl::list<LinearSolver::Tags::Operand<
      NonlinearSolver::Tags::Correction<Tags::ConformalFactor<DataVector>>>>;
  using nonlinear_gradient_tags = tmpl::list<Tags::ConformalFactor<DataVector>>;
  using divergence_tags = tmpl::list<>;
};
}  // namespace Xcts
