// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementActions.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to the %Schwarz linear solver
///
/// \see `LinearSolver::Schwarz::Schwarz`
namespace LinearSolver::Schwarz {

/*!
 * \ingroup LinearSolverGroup
 * \brief An additive Schwarz subdomain solver for linear systems of equations
 * \f$Ax=b\f$.
 *
 * This Schwarz-type linear solver works by solving many sub-problems in
 * parallel and combining their solutions as a weighted sum to converge towards
 * the global solution. Each sub-problem is the restriction of the global
 * problem to an element-centered subdomain, which consists of a central element
 * and an overlap region with its neighbors. The decomposition into independent
 * sub-problems makes this linear solver very parallelizable, avoiding global
 * synchronization points altogether. It is commonly used as a preconditioner to
 * Krylov-type solvers such as `LinearSolver::gmres::Gmres` or
 * `LinearSolver::cg::ConjugateGradient`, or as the "smoother" for a Multigrid
 * solver (which may in turn precondition a Krylov-type solver).
 *
 * This linear solver relies on an implementation of the global linear operator
 * \f$A(x)\f$ and its restriction to a subdomain \f$A_{ss}(x)\f$. Each step of
 * the algorithm expects that \f$A(x)\f$ is computed and stored in the DataBox
 * as `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>`.
 * To perform a solve, add the `solve` action list to an array parallel
 * component. Pass the actions that compute \f$A(x)\f$, as well as any further
 * actions you wish to run in each step of the algorithm, as the first template
 * parameter to `solve`. If you add the `solve` action list multiple times, use
 * the second template parameter to label each solve with a different type.
 * Pass the subdomain operator as the `SubdomainOperator` template parameter to
 * this class (not to the `solve` action list template). See the paragraph below
 * for information on implementing a subdomain operator.
 *
 * \par Subdomain geometry:
 * The image below gives an overview of the structure of an element-centered
 * subdomain:
 *
 * \image html subdomain_structure.svg "Fig. 1: An element-centered subdomain" width=600px
 *
 * Fig. 1 shows part of a 2-dimensional computational domain. The domain is
 * composed of elements (light gray) that each carry a Legendre-Gauss-Lobatto
 * mesh of grid points (black dots). The diagonally shaded region to the left
 * illustrates an external domain boundary. The lines between neighboring
 * element faces illustrate mortar meshes, which are relevant when implementing
 * a subdomain operator in a DG context but play no role in the Schwarz
 * algorithm. The dashed line gives an example of an element-centered subdomain
 * with a maximum of 2 grid points of overlap into neighboring elements. For
 * details on how the number of overlap points is chosen see
 * `LinearSolver::Schwarz::overlap_extent`. Note that the subdomain does not
 * extend into corner- or edge-neighbors, which is different to both
 * \cite Stiller2016a and \cite Vincent2019qpd. The reason we don't include
 * such diagonal couplings is that in a DG context information only propagates
 * across faces, as noted in \cite Stiller2016a. By eliminating the corner-
 * and edge-neighbors we significantly reduce the complexity of the subdomain
 * geometry and potentially also the communication costs. However, the
 * advantages and disadvantages of this choice have yet to be carefully
 * evaluated.
 *
 * \par Subdomain operator:
 * The Schwarz subdomain solver relies on a restriction of the global linear
 * problem \f$Ax=b\f$ to the subdomains. The subdomain operator
 * \f$A_{ss}=R_s A R_s^T\f$, where \f$R_s\f$ is the restriction operator,
 * essentially assumes that its operand is zero everywhere but within the
 * subdomain (see Section 3.1 in \cite Stiller2016a). Therefore it can be
 * evaluated entirely on an element-centered subdomain with no need to
 * communicate with neighbors within each subdomain-operator application, as
 * opposed to the global linear operator that typically requires
 * nearest-neighbor communications. See
 * `LinearSolver::Schwarz::SubdomainOperator` for details on how to
 * implement a subdomain operator for your problem.
 *
 * \par Algorithm overview:
 * In each iteration, the Schwarz algorithm computes the residual \f$r=b-Ax\f$,
 * restricts it to all subdomains as \f$r_s=R_s r\f$ and communicates data on
 * overlap regions with neighboring elements. Once an element has received all
 * data on its subdomain it solves the sub-problem \f$A_{ss}\delta x_s=r_s\f$
 * for the correction \f$\delta x_s\f$, where \f$\delta x_s\f$ and \f$r_s\f$
 * have data only on the points of the subdomain and \f$A_{ss}\f$ is the
 * subdomain operator. Since all elements perform such a subdomain-solve, we end
 * up with a subdomain solution \f$\delta x_s\f$ on every subdomain, and the
 * solutions overlap. Therefore, the algorithm communicates subdomain solutions
 * on overlap regions with neighboring elements and adds them to the solution
 * field \f$x\f$ as a weighted sum.
 *
 * \par Applying the global linear operator:
 * In order to compute the residual \f$r=b-Ax\f$ that will be restricted to the
 * subdomains to serve as source for the subdomain solves, we must apply the
 * global linear operator \f$A\f$ to the solution field \f$x\f$ once per Schwarz
 * iteration. The global linear operator typically introduces nearest-neighbor
 * couplings between elements but no global synchronization point (as opposed to
 * Krylov-type solvers such as `LinearSolver::gmres::Gmres` that require global
 * inner products in each iteration). The algorithm assumes that the action list
 * passed to `solve` applies the global linear operator, as described above.
 *
 * \par Subdomain solves:
 * Each subdomain solve is local to an element, since the data on the subdomain
 * has been made available through communication with the neighboring elements.
 * We can now use any means to solve the subdomain problem that's appropriate
 * for the subdomain operator \f$A_{ss}\f$. For example, if the subdomain
 * operator was available as an explicit matrix we could invert it directly.
 * Since it is typically a matrix-free operator a common approach to solve the
 * subdomain problem is by means of an iterative Krylov-type method, such as
 * GMRES or Conjugate Gradient, ideally with an appropriate preconditioner (yes,
 * this would be preconditioned Krylov-methods solving the subdomains of the
 * Schwarz solver, which might in turn precondition a global Krylov-solver -
 * it's preconditioners all the way down). Currently, the linear solvers listed
 * in `LinearSolver::Serial` are available to solve subdomains. They include
 * Krylov-type methods that support nested preconditioning with any other linear
 * solver. Additional problem-specific subdomain preconditioners can be added
 * with the `SubdomainPreconditioners` template parameter. Note that the choice
 * of subdomain solver (and, by extension, the choice of subdomain
 * preconditioner) affects only the _performance_ of the Schwarz solver, not its
 * convergence or parallelization properties (assuming the subdomain solutions
 * it produces are sufficiently precise).
 *
 * \par Weighting:
 * Once the subdomain solutions \f$\delta x_s\f$ have been found they must be
 * combined where they have multiple values, i.e. on overlap regions of the
 * subdomains. We use an additive approach where we combine the subdomain
 * solutions as a weighted sum, which has the advantage over "multiplicative"
 * Schwarz methods that the subdomains decouple and can be solved in parallel.
 * See `LinearSolver::Schwarz::element_weight` and Section 3.1 of
 * \cite Stiller2016a for details. Note that we must account for the missing
 * corner- and edge-neighbors when constructing the weights. See
 * `LinearSolver::Schwarz::intruding_weight` for a discussion.
 *
 * \par Array sections
 * This linear solver requires no synchronization between elements, so it runs
 * on all elements in the array parallel component. Partitioning of the elements
 * in sections is only relevant for observing residual norms. Pass the section
 * ID tag for the `ArraySectionIdTag` template parameter if residual norms
 * should be computed over a section. Pass `void` (default) to compute residual
 * norms over all elements in the array.
 *
 * \par Possible improvements:
 * - Specify the number of overlap points as a fraction of the element width
 * instead of a fixed number. This was shown in \cite Stiller2016b to achieve
 * scale-independent convergence at the cost of increasing subdomain sizes.
 */
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename SubdomainPreconditioners = tmpl::list<>,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>,
          typename ArraySectionIdTag = void>
struct Schwarz {
  using operand_tag = FieldsTag;
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using options_group = OptionsGroup;

  using component_list = tmpl::list<>;
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<async_solvers::reduction_data, detail::reduction_data>>;
  using subdomain_solver =
      detail::subdomain_solver<FieldsTag, SubdomainOperator,
                               SubdomainPreconditioners>;

  using initialize_element = tmpl::list<
      async_solvers::InitializeElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::InitializeElement<FieldsTag, OptionsGroup, SubdomainOperator,
                                SubdomainPreconditioners>>;

  using amr_projectors = initialize_element;

  using register_element =
      tmpl::list<async_solvers::RegisterElement<FieldsTag, OptionsGroup,
                                                SourceTag, ArraySectionIdTag>,
                 detail::RegisterElement<FieldsTag, OptionsGroup, SourceTag,
                                         ArraySectionIdTag>>;

  template <typename ApplyOperatorActions, typename Label = OptionsGroup>
  using solve = tmpl::list<
      async_solvers::PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label,
                                  ArraySectionIdTag>,
      detail::SendOverlapData<FieldsTag, OptionsGroup, SubdomainOperator>,
      detail::SolveSubdomain<FieldsTag, OptionsGroup, SubdomainOperator,
                             ArraySectionIdTag>,
      detail::ReceiveOverlapSolution<FieldsTag, OptionsGroup,
                                     SubdomainOperator>,
      ApplyOperatorActions,
      async_solvers::CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label,
                                  ArraySectionIdTag>>;
};

}  // namespace LinearSolver::Schwarz
