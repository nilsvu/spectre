// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the linear solver

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Informer/Verbosity.hpp"

/// \cond
namespace NonlinearSolver {
namespace OptionTags {
struct ConvergenceCriteria;
}  // namespace OptionTags
}  // namespace NonlinearSolver
/// \endcond

/*!
 * \ingroup LinearSolverGroup
 * \brief Functionality for solving linear systems of equations
 */
namespace NonlinearSolver {
namespace Tags {

/*
 * \brief The correction \f$\delta x\f$ to improve a solution \f$x_0\f$
 *
 * \details A linear problem \f$Ax=b\f$ can be equivalently formulated as the
 * problem \f$A\delta x=b-A x_0\f$ for the correction \f$\delta x\f$ to an
 * initial guess \f$x_0\f$. More importantly, we can use a correction scheme
 * to solve a nonlinear problem \f$A_\mathrm{nonlinear}(x)=b\f$ by repeatedly
 * solving a linearization of it. For instance, a Newton-Raphson scheme
 * iteratively refines an initial guess \f$x_0\f$ by repeatedly solving the
 * linearized problem
 * \f[
 * \frac{\delta A_\mathrm{nonlinear}}{\delta x}(x_k)\delta x_k =
 * b-A_\mathrm{nonlinear}(x_k)
 * \f]
 * for the correction \f$\delta x_k\f$ and then updating the solution as
 * \f$x_{k+1}=x_k + \delta x_k\f$.
 */
template <typename Tag>
struct Correction : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "Correction(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The nonlinear operator \f$A_\mathrm{nonlinear}\f$ applied to the data
 * in `Tag`
 */
template <typename Tag>
struct OperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Nonlinear" prefix to abbreviate the namespace for uniqueness
    return "NonlinearOperatorAppliedTo(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Identifies a step in the nonlinear solver algorithm
 */
struct IterationId : db::SimpleTag {
  static std::string name() noexcept {
    // Add "Nonlinear" prefix to abbreviate the namespace for uniqueness
    return "NonlinearIterationId";
  }
  using type = size_t;
  template <typename Tag>
  using step_prefix = OperatorAppliedTo<Tag>;
};

/*!
 * \brief Computes the `NonlinearSolver::Tags::IterationId` incremented by one.
 */
struct NextIterationIdCompute : db::ComputeTag, ::Tags::Next<IterationId> {
  using argument_tags = tmpl::list<IterationId>;
  static size_t function(const size_t& iteration_id) noexcept {
    return iteration_id + 1;
  }
};

/*!
 * \brief The nonlinear residual
 * \f$r_\mathrm{nonlinear} = b - A_\mathrm{nonlinear}(\delta x)\f$
 */
template <typename Tag>
struct Residual : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Nonlinear" prefix to abbreviate the namespace for uniqueness
    return "NonlinearResidual(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Holds a `Convergence::HasConverged` flag that signals the nonlinear
 * solver has converged, along with the reason for convergence.
 */
struct HasConverged : db::SimpleTag {
  static std::string name() noexcept { return "NonlinearSolverHasConverged"; }
  using type = Convergence::HasConverged;
};

/*
 * \brief Employs the `NonlinearSolver::OptionTags::ConvergenceCriteria` to
 * determine the nonlinear solver has converged.
 */
template <typename FieldsTag>
struct HasConvergedCompute : NonlinearSolver::Tags::HasConverged,
                             db::ComputeTag {
 private:
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, FieldsTag>>;
  using initial_residual_magnitude_tag =
      db::add_tag_prefix<::Tags::Initial, residual_magnitude_tag>;

 public:
  using argument_tags =
      tmpl::list<NonlinearSolver::OptionTags::ConvergenceCriteria,
                 NonlinearSolver::Tags::IterationId, residual_magnitude_tag,
                 initial_residual_magnitude_tag>;
  static db::item_type<NonlinearSolver::Tags::HasConverged> function(
      const Convergence::Criteria& convergence_criteria,
      const size_t& iteration_id, const double& residual_magnitude,
      const double& initial_residual_magnitude) noexcept {
    return Convergence::HasConverged(convergence_criteria, iteration_id,
                                     residual_magnitude,
                                     initial_residual_magnitude);
  }
};

}  // namespace Tags

namespace OptionTags {

/*!
 * \brief The `::Verbosity` of nonlinear solver logging
 */
struct Verbosity {
  using type = ::Verbosity;
  static constexpr OptionString help = {"Verbosity of the nonlinear solver"};
  static type default_value() noexcept { return ::Verbosity::Quiet; }
  // This needs a unique name since we don't have a grouping feature yet.
  static std::string name() noexcept { return "NonlinearVerbosity"; }
};

/*!
 * \brief `Convergence::Criteria` that determine the nonlinear solve has
 * converged
 */
struct ConvergenceCriteria : db::SimpleTag {
  static constexpr OptionString help =
      "These determine the nonlinear solve has converged";
  using type = Convergence::Criteria;
  // This needs a unique name since we don't have a grouping feature yet
  static std::string name() noexcept { return "NonlinearConvCrit"; }
};

}  // namespace OptionTags
}  // namespace NonlinearSolver
