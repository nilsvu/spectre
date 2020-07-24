// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to the %Schwarz linear solver
///
/// \see `LinearSolver::Schwarz::Schwarz`
namespace LinearSolver::Schwarz {

/*!
 * \ingroup LinearSolverGroup
 * \brief An additive Schwarz subdomain solver for linear systems of equations
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
 * subdomain (see Section 3.1 in \cite Stiller2016a). See
 * `LinearSolver::Schwarz::protocols::SubdomainOperator` for details on how to
 * implement a subdomain operator for your problem.
 *
 * \par Algorithm overview:
 * In each iteration, the Schwarz algorithm restricts the residual \f$r=b-Ax\f$
 * to all subdomains and communicates data on overlap regions with neighboring
 * elements. Once an element has received all data on its subdomain it solves
 * the sub-problem \f$A_{ss}\delta x_s=r_s\f$ for the correction \f$\delta
 * x_s\f$, where \f$\delta x_s\f$ and \f$r_s\f$ have data only on the points of
 * the subdomain and \f$A_{ss}\f$ is the subdomain operator. Since all elements
 * perform such a subdomain-solve, we end up with a subdomain solution \f$\delta
 * x_s\f$ on every subdomain, and the solutions overlap. Therefore, the
 * algorithm communicates subdomain solutions on overlap regions with
 * neighboring elements and adds them to the solution field \f$x\f$ as a
 * weighted sum.
 *
 * \par Subdomain solves:
 * Each subdomain solve is local to an element, since the data on the subdomain
 * has been made available through communication with the neighboring elements.
 * We can now use any means to solve the subdomain problem that's appropriate
 * for the subdomain operator \f$A_{ss}\f$. For example, if the subdomain
 * operator was available as an explit matrix we could invert it directly. Since
 * it is typically a matrix-free operator a common approach to solve the
 * subdomain problem is by means of an iterative Krylov-type method, such as
 * GMRES or Conjugate Gradient, ideally with an appropriate preconditioner (yes,
 * this would be preconditioned Krylov-methods solving the subdomains of the
 * Schwarz solver, which might in turn precondition a global Krylov-solver -
 * it's preconditioners all the way down). Currently, we use
 * `LinearSolver::Serial::Gmres` to solve subdomains. It supports
 * preconditioning, and adding useful subdomain preconditioners will be the
 * subject of future work. Note that the choice of subdomain solver (and, by
 * extension, the choice of subdomain preconditioner) affects only the
 * _performance_ of the Schwarz solver, not its convergence or parallelization
 * properties (assuming the subdomain solutions it produces are sufficiently
 * precise).
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
 * \par Possible improvements:
 * - Specify the number of overlap points as a fraction of the element width
 * instead of a fixed number. This was shown in \cite Stiller2016b to achieve
 * scale-independent convergence at the cost of increasing subdomain sizes.
 * - Subdomain preconditioning (see paragraph on subdomain solves above)
 */
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename SubdomainPreconditioner = void,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Schwarz {
  using operand_tag = FieldsTag;
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using options_group = OptionsGroup;
  static_assert(
      tt::assert_conforms_to<SubdomainOperator, protocols::SubdomainOperator>);

  using component_list = tmpl::list<>;
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<async_solvers::reduction_data, detail::reduction_data>>;

  using initialize_element = tmpl::list<
      async_solvers::InitializeElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::InitializeElement<FieldsTag, OptionsGroup, SubdomainOperator,
                                SubdomainPreconditioner>>;

  using register_element = tmpl::list<
      async_solvers::RegisterElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::RegisterElement<FieldsTag, OptionsGroup, SourceTag>>;

  template <typename ApplyOperatorActions, typename Label = OptionsGroup>
  using solve = tmpl::list<
      async_solvers::PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label>,
      detail::SendOverlapData<FieldsTag, OptionsGroup, SubdomainOperator>,
      detail::ReceiveOverlapData<FieldsTag, OptionsGroup, SubdomainOperator>,
      detail::SolveSubdomain<FieldsTag, OptionsGroup, SubdomainOperator,
                             SubdomainPreconditioner>,
      detail::ReceiveOverlapSolution<FieldsTag, OptionsGroup,
                                     SubdomainOperator>,
      ApplyOperatorActions,
      async_solvers::CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label>>;
};

}  // namespace LinearSolver::Schwarz
