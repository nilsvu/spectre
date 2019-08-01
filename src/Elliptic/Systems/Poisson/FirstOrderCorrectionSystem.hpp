// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson {

template <size_t Dim>
struct LinearizedFirstOrderCorrectionSystem;

template <size_t Dim>
struct FirstOrderCorrectionSystem {
 private:
  using field = Tags::Field;
  using field_gradient =
      ::Tags::deriv<field, tmpl::size_t<Dim>, Frame::Inertial>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using primal_fields = tmpl::list<field>;
  using auxiliary_fields = tmpl::list<field_gradient>;
  using fields_tag =
      ::Tags::Variables<tmpl::append<primal_fields, auxiliary_fields>>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag = fields_tag;
  using primal_variables = primal_fields;
  using auxiliary_variables = auxiliary_fields;

  using fluxes = EuclideanFluxes<Dim>;
  using sources = Sources;

  using linearized_system = LinearizedFirstOrderCorrectionSystem<Dim>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim>
struct LinearizedFirstOrderCorrectionSystem {
 private:
  using original_system = FirstOrderCorrectionSystem<Dim>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using primal_fields =
      db::wrap_tags_in<NonlinearSolver::Tags::Correction,
                       typename original_system::primal_fields>;
  using auxiliary_fields =
      db::wrap_tags_in<NonlinearSolver::Tags::Correction,
                       typename original_system::auxiliary_fields>;
  using fields_tag = db::add_tag_prefix<NonlinearSolver::Tags::Correction,
                                        typename original_system::fields_tag>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using primal_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, primal_fields>;
  using auxiliary_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, auxiliary_fields>;

  using fluxes = typename original_system::fluxes;
  using sources = typename original_system::sources;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

}  // namespace Poisson
