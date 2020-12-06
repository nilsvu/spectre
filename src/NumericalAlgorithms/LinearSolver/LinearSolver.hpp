// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"

namespace LinearSolver::Serial {

/// Registrars for linear solvers
namespace Registrars {}

namespace detail {
DEFINE_FAKE_VIRTUAL(solve);
}  // namespace detail

/// Base class for serial linear solvers
template <typename LinearSolverRegistrars>
class LinearSolver : public PUP::able {
 protected:
  /// \cond
  LinearSolver() = default;
  LinearSolver(const LinearSolver&) = default;
  LinearSolver(LinearSolver&&) = default;
  LinearSolver& operator=(const LinearSolver&) = default;
  LinearSolver& operator=(LinearSolver&&) = default;
  /// \endcond

 public:
  virtual ~LinearSolver() = default;

  WRAPPED_PUPable_abstract(LinearSolver);  // NOLINT

  using creatable_classes = Registration::registrants<LinearSolverRegistrars>;

  using Inherit = detail::FakeVirtualInherit_solve<LinearSolver>;

  /*!
   * \brief Solve the linear equation \f$Ax=b\f$ where \$fA\f$ is the
   * `linear_operator` and \f$b\f$ is the `source`.
   *
   * - The (approximate) solution \f$x\f$ is returned in the
   *   `initial_guess_in_solution_out` buffer, which also serves to provide an
   *   initial guess for \f$x\f$. Not all solvers take the initial guess into
   *   account, but expect the buffer is sized correctly.
   * - The `linear_operator` must be an invocable that takes a `VarsType` as
   *   const-ref argument and returns a `SourceType` by reference.
   *
   * Each solve can mutate the private state of the solver, for example to cache
   * quantities to accelerate successive solves for the same operator. Invoke
   * `reset` to discard these caches.
   */
  template <typename LinearOperator, typename VarsType, typename SourceType,
            typename... Args>
  Convergence::HasConverged solve(
      const gsl::not_null<VarsType*> initial_guess_in_solution_out,
      LinearOperator&& linear_operator, const SourceType& source,
      Args&&... args) const noexcept {
    return detail::fake_virtual_solve<creatable_classes>(
        this, initial_guess_in_solution_out,
        std::forward<LinearOperator>(linear_operator), source,
        std::forward<Args>(args)...);
  }

  /// Discard caches from previous solves. Use when the linear operator you are
  /// solving changed.
  virtual void reset() noexcept = 0;
};

}  // namespace LinearSolver::Serial
