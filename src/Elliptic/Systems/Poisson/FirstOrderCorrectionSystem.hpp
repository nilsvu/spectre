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
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
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

template <size_t Dim>
struct LinearizedFirstOrderCorrectionSystem {
  static constexpr size_t volume_dim = Dim;

  using fields_tag = db::add_tag_prefix<
      NonlinearSolver::Tags::Correction,
      Tags::Variables<tmpl::list<Field, AuxiliaryField<Dim>>>>;

  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;

  using compute_fluxes = ComputeFirstOrderFluxes<
      Dim, variables_tag,
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<Field>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<AuxiliaryField<Dim>>>>;
  using compute_sources = ComputeFirstOrderSources<
      Dim, variables_tag,
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<Field>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<AuxiliaryField<Dim>>>>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using primal_variables = tmpl::list<
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<Field>>>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim>
struct FirstOrderCorrectionSystem {
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag = Tags::Variables<tmpl::list<Field, AuxiliaryField<Dim>>>;

  using compute_fluxes =
      ComputeFirstOrderFluxes<Dim, fields_tag, Field, AuxiliaryField<Dim>>;
  using compute_sources =
      ComputeFirstOrderSources<Dim, fields_tag, Field, AuxiliaryField<Dim>>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using primal_variables = tmpl::list<Field>;
  using compute_analytic_fluxes = ComputeFirstOrderFluxes<
      Dim, db::add_tag_prefix<::Tags::Analytic, fields_tag>,
      ::Tags::Analytic<Field>, ::Tags::Analytic<AuxiliaryField<Dim>>>;

  using linearized_system = LinearizedFirstOrderCorrectionSystem<Dim>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

}  // namespace Poisson
