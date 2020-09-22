// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ElementActions.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitor.hpp"
#include "Utilities/TMPL.hpp"

namespace NonlinearSolver::newton_raphson {

/*!
 * \brief A Newton-Raphson correction scheme for nonlinear systems of equations
 * \f$A_\mathrm{nonlinear}(x)=b\f$.
 *
 * \details We can use a correction scheme to solve a nonlinear problem
 * \f$A_\mathrm{nonlinear}(x)=b\f$ by repeatedly solving a linearization of it.
 * A Newton-Raphson scheme iteratively refines an initial guess \f$x_0\f$ by
 * repeatedly solving the linearized problem \f[\frac{\delta
 * A_\mathrm{nonlinear}}{\delta x}(x_k) \delta x_k = b-A_\mathrm{nonlinear}(x_k)
 * \f] for the correction \f$\delta x_k\f$ and then updating the solution as
 * \f$x_{k+1}=x_k + \delta x_k\f$.
 *
 * The only operation we need to supply to the algorithm is the result of the
 * operation \f$A_\mathrm{nonlinear}(x)\f$. Each step of the algorithm expects
 * that \f$A_\mathrm{nonlinear}(x)\f$ has been computed in a preceding action
 * and stored in the DataBox as
 * %db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, FieldsTag>.
 *
 * \par Globalization:
 * This nonlinear solver supports a line-search globalization.
 */
template <typename Metavariables, typename FieldsTag, typename OptionsGroup,
          typename ArraySectionIdTag = void>
struct NewtonRaphson {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;

  using operand_tag = fields_tag;
  using linear_solver_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using linear_solver_source_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;

  /// The parallel components used by the nonlinear solver
  using component_list = tmpl::list<
      detail::ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>;

  using initialize_element = detail::InitializeElement<FieldsTag, OptionsGroup>;

  using register_element = tmpl::list<>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<LinearSolver::observe_detail::reduction_data>>;

  template <typename ApplyNonlinearOperator, typename SolveLinearization,
            typename Label = OptionsGroup>
  using solve = tmpl::list<
      detail::PrepareSolve<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>,
      detail::InitializeHasConverged<FieldsTag, OptionsGroup, Label,
                                     ArraySectionIdTag>,
      detail::PrepareStep<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>,
      SolveLinearization,
      detail::PerformStep<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>,
      ApplyNonlinearOperator,
      detail::ContributeToResidualMagnitudeReduction<FieldsTag, OptionsGroup,
                                                     Label, ArraySectionIdTag>,
      detail::GlobalizeAndCompleteStep<FieldsTag, OptionsGroup, Label,
                                       ArraySectionIdTag>>;
};

}  // namespace NonlinearSolver::newton_raphson
