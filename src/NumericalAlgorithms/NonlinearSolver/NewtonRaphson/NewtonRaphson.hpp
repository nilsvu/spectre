// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "NumericalAlgorithms/NonlinearSolver/NewtonRaphson/ElementActions.hpp"
#include "NumericalAlgorithms/NonlinearSolver/NewtonRaphson/InitializeElement.hpp"
#include "NumericalAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitor.hpp"
#include "Utilities/TMPL.hpp"

namespace NonlinearSolver {

/*!
 * \brief A Newton-Raphson correction scheme for nonlinear systems of equations
 * \f$A_\mathrm{nonlinear}(x)=b\f$.
 *
 * \details We can use a correction scheme to solve a nonlinear problem
 * \f$A_\mathrm{nonlinear}(x)=b\f$ by repeatedly solving a linearization of it.
 * A Newton-Raphson scheme iteratively refines an initial guess \f$x_0\f$ by
 * repeatedly solving the linearized problem
 * \f[
 * \frac{\delta A_\mathrm{nonlinear}}{\delta x}(x_k) \delta x_k
 * = b-A_\mathrm{nonlinear}(x_k)
 * \f]
 * for the correction \f$\delta x_k\f$ and then updating the solution as
 * \f$x_{k+1}=x_k + \delta x_k\f$.
 *
 * \note We plan to add a line search to this scheme to improve its convergence
 * properties.
 *
 * The only operation we need to supply to the algorithm is the
 * result of the operation \f$A_\mathrm{nonlinear}(x)\f$. Each invocation
 * of the `prepare_linear_solve` action expects that
 * \f$A_\mathrm{nonlinear}(x)\f$ has been computed in a preceding action and
 * stored in the DataBox as
 * %db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo,
 * typename Metavariables::system::nonlinear_fields_tag>.
 */
template <typename Metavariables>
struct NewtonRaphson {
  /*!
   * \brief The parallel components used by the nonlinear solver
   *
   * Uses:
   * - System:
   *   * `nonlinear_fields_tag`
   */
  using component_list =
      tmpl::list<newton_raphson_detail::ResidualMonitor<Metavariables>>;

  /*!
   * \brief Initialize the tags used by the nonlinear solver
   *
   * Uses:
   * - System:
   *   * `nonlinear_fields_tag`
   *
   * With:
   * - `operand_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>`
   * - `operator_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>`
   * - `residual_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>`
   *
   * DataBox changes:
   * - Adds:
   *   * `NonlinearSolver::Tags::IterationId`
   *   * `Tags::Next<NonlinearSolver::Tags::IterationId`
   *   * `linear_source_tag`
   *   * `linear_operator_tag`
   *   * `NonlinearSolver::Tags::HasConverged`
   * - Removes: nothing
   * - Modifies:
   *   * `correction_tag`
   *
   * \note The `correction_tag` must already be present in the DataBox and is
   * set to its initial value here. It is typically added to the DataBox by the
   * system, which uses it to compute the `operator_tag` in each step. Also the
   * `operator_tag` is typically added to the DataBox by the system, but does
   * not need to be initialized until it is computed for the first time in the
   * first step of the algorithm.
   */
  using tags = newton_raphson_detail::InitializeElement<Metavariables>;

  // Compile-time interface for observers
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<observe_detail::reduction_data>>;

  using prepare_linear_solve = newton_raphson_detail::PrepareLinearSolve;
  using perform_step = newton_raphson_detail::PerformStep;
};

}  // namespace NonlinearSolver
