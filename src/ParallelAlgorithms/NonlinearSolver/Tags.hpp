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
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

namespace NonlinearSolver {

namespace OptionTags {

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the iterative nonlinear solver, e.g.
 * convergence criteria.
 */
struct Group {
  static std::string name() noexcept { return "NonlinearSolver"; }
  static constexpr OptionString help =
      "Options for the iterative nonlinear solver";
};

struct ConvergenceCriteria {
  static constexpr OptionString help =
      "Determine convergence of the nonlinear solve";
  using type = Convergence::Criteria;
  using group = Group;
};

struct Verbosity {
  using type = ::Verbosity;
  static constexpr OptionString help = {"Logging verbosity"};
  static type default_value() noexcept { return ::Verbosity::Quiet; }
  using group = Group;
};

}  // namespace OptionTags

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

struct StepLength : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "StepLength"; }
};

/*!
 * \brief Holds a flag that signals the globalization has converged.
 */
struct GlobalizationHasConverged : db::SimpleTag {
  static std::string name() noexcept {
    return "NonlinearGlobalizationHasConverged";
  }
  using type = bool;
};

struct GlobalizationIterationId : db::SimpleTag {
  using type = size_t;
  static std::string name() noexcept { return "GlobalizationIterationId"; }
};

struct GlobalizationIterationsHistory : db::SimpleTag {
  using type = std::vector<size_t>;
  static std::string name() noexcept {
    return "GlobalizationIterationsHistory";
  }
};

struct TemporalId : db::SimpleTag {
  using type = size_t;
  static std::string name() noexcept { return "TemporalId"; }
  template <typename Tag>
  using step_prefix = OperatorAppliedTo<Tag>;
};

struct TemporalIdCompute : db::ComputeTag, TemporalId {
  using argument_tags =
      tmpl::list<GlobalizationIterationId, GlobalizationIterationsHistory>;
  static size_t function(
      const size_t& globalization_iteration_id,
      const std::vector<size_t>& globalization_iterations_history) {
    return alg::accumulate(globalization_iterations_history, size_t{0},
                           funcl::Plus<>{}) +
           globalization_iteration_id;
  }
};
struct TemporalIdNextCompute : db::ComputeTag, ::Tags::Next<TemporalId> {
  using argument_tags = tmpl::list<TemporalId>;
  static size_t function(const size_t& temporal_id) { return temporal_id + 1; }
};

/*!
 * \brief Holds a `Convergence::HasConverged` flag that signals the nonlinear
 * solver has converged, along with the reason for convergence.
 */
struct HasConverged : db::SimpleTag {
  static std::string name() noexcept { return "NonlinearSolverHasConverged"; }
  using type = Convergence::HasConverged;
};

/*!
 * \brief `Convergence::Criteria` that determine the nonlinear solve has
 * converged
 *
 * \see NonlinearSolver::OptionTags::ConvergenceCriteria
 */
struct ConvergenceCriteria : db::SimpleTag {
  static std::string name() noexcept {
    return "NonlinearSolverConvergenceCriteria";
  }
  using type = Convergence::Criteria;
  using option_tags = tmpl::list<OptionTags::ConvergenceCriteria>;
  static type create_from_options(const type& option) { return option; }
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
      tmpl::list<NonlinearSolver::Tags::ConvergenceCriteria,
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

/*!
 * \brief Logging verbosity of the nonlinear solver
 *
 * \see NonlinearSolver::OptionTags::Verbosity
 */
struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;
  static std::string name() noexcept { return "NonlinearSolverVerbosity"; }
  using option_tags = tmpl::list<OptionTags::Verbosity>;
  static type create_from_options(const type& option) { return option; }
};

}  // namespace Tags

}  // namespace NonlinearSolver

namespace Tags {

template <>
struct NextCompute<NonlinearSolver::Tags::IterationId>
    : Next<NonlinearSolver::Tags::IterationId>, db::ComputeTag {
  using argument_tags = tmpl::list<NonlinearSolver::Tags::IterationId>;
  static size_t function(const size_t& iteration_id) noexcept {
    return iteration_id + 1;
  }
};

}  // namespace Tags
