// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Serial {

/// \cond
template <typename LinearSolverRegistrars>
struct ExplicitInverse;
/// \endcond

namespace Registrars {
struct ExplicitInverse {
  template <typename RegistrarList>
  using f = Serial::ExplicitInverse<RegistrarList>;
};
}  // namespace Registrars

/*!
 * \brief Linear solver that builds a matrix representation of the linear
 * operator and inverts it directly
 *
 * This solver first constructs an explicit matrix representation by "sniffing
 * out" the operator, i.e. feeding it with unit vectors, and then directly
 * inverts the matrix. The result is an operator that solves the linear problem
 * in a single step. This means that each element has a large initialization
 * cost, but all successive solves converge immediately.
 *
 * \par Advice on using this linear solver:
 *
 * - This solver is entirely agnostic to the structure of the linear operator.
 *   Be advised to implement a linear solver that is specialized for your linear
 *   operator to take advantage of its properties, e.g. its tensor-product
 *   structure. Only use this solver if no alternatives are available and if you
 *   have verified that it speeds up your solves.
 * - Since this linear solver stores the full inverse operator matrix it can
 *   have significant memory demands. For example, an operator representing a 3D
 *   first-order Elasticity system (9 variables) discretized on 12 grid points
 *   per dimension requires ca. 2GB of memory (per element) to store the matrix,
 *   scaling quadratically with the number of variables and with a power of 6
 *   with the number of grid points per dimension. Therefore, be advised to
 *   distribute the elements on a sufficient number of nodes to meet the memory
 *   requirements.
 * - This linear solver can be `reset()` when the operator changes (e.g. in each
 *   nonlinear-solver iteration). However, when using this solver as
 *   preconditioner it can be advantageous to avoid the reset and the
 *   corresponding cost of re-building the matrix and its inverse if the
 *   operator only changes "a little". In that case the preconditioner solves
 *   subdomain problems only approximately, but possibly still sufficiently to
 *   provide effective preconditioning. You can toggle this behaviour with the
 *   `DisableResets` option.
 */
template <typename LinearSolverRegistrars =
              tmpl::list<Registrars::ExplicitInverse>>
class ExplicitInverse : public LinearSolver<LinearSolverRegistrars>::Inherit {
 private:
  struct DisableResets {
    using type = bool;
    static constexpr Options::String help =
        "Enable or disable resets. This only has an effect in cases where the "
        "operator changes, e.g. between nonlinear-solver iterations. Only "
        "disable resets when using this solver as preconditioner and thus "
        "approximate solutions are desirable. Disabling resets avoids "
        "expensive re-building of the operator, but comes at the cost of less "
        "accurate preconditioning and thus potentially more preconditioned "
        "iterations. Whether or not this helps convergence overall is highly "
        "problem-dependent.";
  };

 public:
  using options = tmpl::list<DisableResets>;
  static constexpr Options::String help =
      "Builds a matrix representation of the subdomain operator and inverts it "
      "directly. This means that each element has a large initialization cost, "
      "but all subdomain solves converge immediately.";

  ExplicitInverse() = default;
  ExplicitInverse(const ExplicitInverse& /*rhs*/) = default;
  ExplicitInverse& operator=(const ExplicitInverse& /*rhs*/) = default;
  ExplicitInverse(ExplicitInverse&& /*rhs*/) = default;
  ExplicitInverse& operator=(ExplicitInverse&& /*rhs*/) = default;
  ~ExplicitInverse() = default;

  explicit ExplicitInverse(bool disable_resets) noexcept
      : disable_resets_(disable_resets) {}

  /// \cond
  explicit ExplicitInverse(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ExplicitInverse);  // NOLINT
  /// \endcond

  /// Solve the equation \f$Ax=b\f$ by explicitly constructing the operator
  /// matrix \f$A\f$ and its inverse. The first solve is computationally
  /// expensive and successive solves are cheap.
  template <typename SubdomainOperator, typename VarsType, typename SourceType>
  Convergence::HasConverged solve(const gsl::not_null<VarsType*>& solution,
                                  SubdomainOperator&& linear_operator,
                                  const SourceType& source) const noexcept;

  /// Flags the operator to require re-initialization. No memory is released.
  /// Call this function to rebuild the solver when the operator changed.
  void reset() noexcept override {
    if (disable_resets_) {
      return;
    }
    size_ = std::numeric_limits<size_t>::max();
  }

  /// Size of the operator. The stored matrix will have `size^2` entries.
  size_t size() const noexcept { return size_; }

  /// The matrix representation of the solver. This matrix approximates the
  /// inverse of the subdomain operator.
  DenseMatrix<double> matrix_representation() const noexcept {
    return inverse_;
  }

  void pup(PUP::er& p) noexcept override {  // NOLINT
    p | disable_resets_;
    p | size_;
    p | inverse_;
    if (p.isUnpacking() and size_ != std::numeric_limits<size_t>::max()) {
      source_workspace_.resize(size_);
      solution_workspace_.resize(size_);
    }
  }

 private:
  bool disable_resets_{};

  // Caches for successive solves of the same operator
  mutable size_t size_ = std::numeric_limits<size_t>::max();
  mutable DenseMatrix<double, blaze::columnMajor> inverse_{};

  // Buffers to avoid re-allocating memory for applying the operator
  mutable DenseVector<double> source_workspace_{};
  mutable DenseVector<double> solution_workspace_{};
};

/// \cond
template <typename LinearSolverRegistrars>
// NOLINTNEXTLINE
PUP::able::PUP_ID ExplicitInverse<LinearSolverRegistrars>::my_PUP_ID = 0;
/// \endcond

template <typename LinearSolverRegistrars>
template <typename SubdomainOperator, typename VarsType, typename SourceType>
Convergence::HasConverged ExplicitInverse<LinearSolverRegistrars>::solve(
    const gsl::not_null<VarsType*>& solution,
    SubdomainOperator&& linear_operator,
    const SourceType& source) const noexcept {
  if (UNLIKELY(size_ == std::numeric_limits<size_t>::max())) {
    const auto& used_for_size = source;
    size_ = used_for_size.size();
    source_workspace_.resize(size_);
    solution_workspace_.resize(size_);
    inverse_.resize(size_, size_);
    // Construct explicit matrix representation by "sniffing out" the operator,
    // i.e. feeding it unit vectors
    auto unit_vector = make_with_value<VarsType>(used_for_size, 0.);
    auto result_buffer = make_with_value<SourceType>(used_for_size, 0.);
    auto& operator_matrix = inverse_;
    size_t i = 0;
    // Re-using the iterators for all operator invocations
    auto result_iterator_begin = result_buffer.begin();
    const auto result_iterator_end = result_buffer.end();
    for (double& unit_vector_data : unit_vector) {
      // Add a 1 at the unit vector location i
      unit_vector_data = 1.;
      // Invoke the operator on the unit vector
      linear_operator(make_not_null(&result_buffer), unit_vector);
      // Set the unit vector back to zero
      unit_vector_data = 0.;
      // Store the result in column i of the matrix
      result_iterator_begin.reset();
      std::copy(result_iterator_begin, result_iterator_end,
                column(operator_matrix, i).begin());
      ++i;
    }
    // Directly invert the matrix
    try {
      blaze::invert(inverse_);
    } catch (const std::invalid_argument& e) {
      ERROR("Could not invert subdomain matrix (size " << size_
                                                       << "): " << e.what());
    }
  }
  // Copy source into contiguous workspace
  std::copy(source.begin(), source.end(), source_workspace_.begin());
  // Apply inverse
  solution_workspace_ = inverse_ * source_workspace_;
  // Reconstruct subdomain data from contiguous workspace
  std::copy(solution_workspace_.begin(), solution_workspace_.end(),
            solution->begin());
  return {0, 0};
}

}  // namespace LinearSolver::Serial
