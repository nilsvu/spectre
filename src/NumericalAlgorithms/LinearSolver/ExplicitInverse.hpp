// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "DataStructures/DynamicMatrix.hpp"
#include "DataStructures/DynamicVector.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/BuildMatrix.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Serial {

/// \cond
template <typename LinearSolverRegistrars>
struct ExplicitInverse;
/// \endcond

namespace Registrars {
/// Registers the `LinearSolver::Serial::ExplicitInverse` linear solver
using ExplicitInverse = Registration::Registrar<Serial::ExplicitInverse>;
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
 *   It is usually better to implement a linear solver that is specialized for
 *   your linear operator to take advantage of its properties. For example, if
 *   the operator has a tensor-product structure, the linear solver might take
 *   advantage of that. Only use this solver if no alternatives are available
 *   and if you have verified that it speeds up your solves.
 * - Since this linear solver stores the full inverse operator matrix it can
 *   have significant memory demands. For example, an operator representing a 3D
 *   first-order Elasticity system (9 variables) discretized on 12 grid points
 *   per dimension requires ca. 2GB of memory (per element) to store the matrix,
 *   scaling quadratically with the number of variables and with a power of 6
 *   with the number of grid points per dimension. Therefore, make sure to
 *   distribute the elements on a sufficient number of nodes to meet the memory
 *   requirements.
 * - This linear solver can be `reset()` when the operator changes (e.g. in each
 *   nonlinear-solver iteration). However, when using this solver as
 *   preconditioner it can be advantageous to avoid the reset and the
 *   corresponding cost of re-building the matrix and its inverse if the
 *   operator only changes "a little". In that case the preconditioner solves
 *   subdomain problems only approximately, but possibly still sufficiently to
 *   provide effective preconditioning.
 */
template <typename LinearSolverRegistrars =
              tmpl::list<Registrars::ExplicitInverse>>
class ExplicitInverse : public LinearSolver<LinearSolverRegistrars> {
 private:
  using Base = LinearSolver<LinearSolverRegistrars>;

 public:
  struct FillFactor {
    using type = size_t;
    static constexpr Options::String help =
        "Compute an incomplete LU factorization of the operator matrix that "
        "has an approximate fill-in of this factor times the fill-in of the "
        "operator matrix. For example, if 10% of the entries in the operator "
        "matrix are non-zero, then a fill-factor of 2 means that ~20% of the "
        "entries in the LU-factorization are non-zero. A reasonable "
        "fill-factor is problem-dependent. A large factor means the "
        "LU-factorization approximates the operator matrix better, hence "
        "reducing the number of iterations if this solver is used as a "
        "preconditioner. A small factor means the LU-factorization is "
        "more sparse, hence reducing the cost to apply this solver. Often, "
        "even a fill-factor of only 1 provides sufficient preconditioning and "
        "minimizes the cost to apply the preconditioner.";
  };
  struct Verbosity {
    using type = ::Verbosity;
    static constexpr Options::String help = "Logging verbosity";
  };
  using options = tmpl::list<FillFactor, Verbosity>;
  static constexpr Options::String help =
      "Build a matrix representation of the linear operator and invert it "
      "directly. This means that the first solve has a large initialization "
      "cost, but all subsequent solves converge immediately.";

  ExplicitInverse() = default;
  explicit ExplicitInverse(const size_t fillin, const ::Verbosity verbosity)
      : fillin_(fillin), verbosity_(verbosity) {}
  ExplicitInverse(const ExplicitInverse& rhs) : Base(rhs) {
    fillin_ = rhs.fillin_;
    verbosity_ = rhs.verbosity_;
    size_ = rhs.size_;
    // TODO: copy ILU
    if (size_ != std::numeric_limits<size_t>::max()) {
      source_workspace_.resize(static_cast<Eigen::Index>(size_));
      solution_workspace_.resize(static_cast<Eigen::Index>(size_));
    }
  };
  ExplicitInverse& operator=(const ExplicitInverse& rhs) {
    Base::operator=(rhs);
    fillin_ = rhs.fillin_;
    verbosity_ = rhs.verbosity_;
    size_ = rhs.size_;
    // TODO: copy ILU
    if (size_ != std::numeric_limits<size_t>::max()) {
      source_workspace_.resize(static_cast<Eigen::Index>(size_));
      solution_workspace_.resize(static_cast<Eigen::Index>(size_));
    }
    return *this;
  };
  ExplicitInverse(ExplicitInverse&& rhs) : Base(rhs) {
    fillin_ = rhs.fillin_;
    verbosity_ = rhs.verbosity_;
    size_ = rhs.size_;
    // TODO: copy ILU
    if (size_ != std::numeric_limits<size_t>::max()) {
      source_workspace_.resize(static_cast<Eigen::Index>(size_));
      solution_workspace_.resize(static_cast<Eigen::Index>(size_));
    }
  };
  ExplicitInverse& operator=(ExplicitInverse&& rhs) {
    Base::operator=(rhs);
    fillin_ = rhs.fillin_;
    verbosity_ = rhs.verbosity_;
    size_ = rhs.size_;
    // TODO: copy ILU
    if (size_ != std::numeric_limits<size_t>::max()) {
      source_workspace_.resize(static_cast<Eigen::Index>(size_));
      solution_workspace_.resize(static_cast<Eigen::Index>(size_));
    }
    return *this;
  };
  ~ExplicitInverse() = default;

  /// \cond
  explicit ExplicitInverse(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ExplicitInverse);  // NOLINT
  /// \endcond

  /*!
   * \brief Solve the equation \f$Ax=b\f$ by explicitly constructing the
   * operator matrix \f$A\f$ and its inverse. The first solve is computationally
   * expensive and successive solves are cheap.
   *
   * Building a matrix representation of the `linear_operator` requires
   * iterating over the `SourceType` (which is also the type returned by the
   * `linear_operator`) in a consistent way. This can be non-trivial for
   * heterogeneous data structures because it requires they define a data
   * ordering. Specifically, the `SourceType` must have a `size()` function as
   * well as `begin()` and `end()` iterators that point into the data. If the
   * iterators have a `reset()` function it is used to avoid repeatedly
   * re-creating the `begin()` iterator. The `reset()` function must not
   * invalidate the `end()` iterator.
   */
  template <typename LinearOperator, typename VarsType, typename SourceType,
            typename... OperatorArgs>
  Convergence::HasConverged solve(
      gsl::not_null<VarsType*> solution, const LinearOperator& linear_operator,
      const SourceType& source,
      const std::tuple<OperatorArgs...>& operator_args = std::tuple{}) const;

  /// Flags the operator to require re-initialization. No memory is released.
  /// Call this function to rebuild the solver when the operator changed.
  void reset() override { size_ = std::numeric_limits<size_t>::max(); }

  /// Size of the operator. The stored matrix will have `size^2` entries.
  size_t size() const { return size_; }

  /// The matrix representation of the solver. This matrix approximates the
  /// inverse of the subdomain operator.
  // const blaze::DynamicMatrix<double, blaze::columnMajor>&
  // matrix_representation() const {
  //   return inverse_;
  // }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    p | fillin_;
    p | verbosity_;
    p | size_;
    // TODO: serialize ILU
    // ilu_.compute(operator_matrix_);
    if (p.isUnpacking() and size_ != std::numeric_limits<size_t>::max()) {
      source_workspace_.resize(static_cast<Eigen::Index>(size_));
      solution_workspace_.resize(static_cast<Eigen::Index>(size_));
    }
  }

  std::unique_ptr<Base> get_clone() const override {
    return std::make_unique<ExplicitInverse>(*this);
  }

 private:
  size_t fillin_ = std::numeric_limits<size_t>::max();
  ::Verbosity verbosity_ = ::Verbosity::Silent;

  // Caches for successive solves of the same operator
  // NOLINTNEXTLINE(spectre-mutable)
  mutable size_t size_ = std::numeric_limits<size_t>::max();
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Eigen::IncompleteLUT<double> ilu_{};

  // Buffers to avoid re-allocating memory for applying the operator
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Eigen::VectorXd source_workspace_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Eigen::VectorXd solution_workspace_{};
};

template <typename LinearSolverRegistrars>
template <typename LinearOperator, typename VarsType, typename SourceType,
          typename... OperatorArgs>
Convergence::HasConverged ExplicitInverse<LinearSolverRegistrars>::solve(
    const gsl::not_null<VarsType*> solution,
    const LinearOperator& linear_operator, const SourceType& source,
    const std::tuple<OperatorArgs...>& operator_args) const {
  if (UNLIKELY(size_ == std::numeric_limits<size_t>::max())) {
    const auto& used_for_size = source;
    size_ = used_for_size.size();
    source_workspace_.resize(static_cast<Eigen::Index>(size_));
    solution_workspace_.resize(static_cast<Eigen::Index>(size_));
    // operator_matrix_.resize(size_, size_);
    // Construct explicit matrix representation by "sniffing out" the operator,
    // i.e. feeding it unit vectors
    auto operand_buffer = make_with_value<VarsType>(used_for_size, 0.);
    auto result_buffer = make_with_value<SourceType>(used_for_size, 0.);
    Eigen::SparseMatrix<double> operator_matrix{
        static_cast<Eigen::Index>(size_), static_cast<Eigen::Index>(size_)};
    build_matrix(make_not_null(&operator_matrix),
                 make_not_null(&operand_buffer), make_not_null(&result_buffer),
                 linear_operator, operator_args);
    // Compute ILU factorization
    if (verbosity_ >= ::Verbosity::Debug) {
      const double matrix_fillin =
          static_cast<double>(operator_matrix.nonZeros()) /
          static_cast<double>(square(size_));
      Parallel::printf("Fillin: %f\n", matrix_fillin);
    }
    ilu_.setFillfactor(fillin_);
    ilu_.compute(operator_matrix);
    // We could free the operator matrix at this point, if we could serialize
    // and copy the ILU class directly.
  }
  // Copy source into contiguous workspace. In cases where the source and
  // solution data are already stored contiguously we might avoid the copy and
  // the associated workspace memory. However, compared to the cost of building
  // and storing the matrix this is likely insignificant.
  std::copy(source.begin(), source.end(), source_workspace_.begin());
  // Apply (approximate) inverse
  solution_workspace_ = ilu_.solve(source_workspace_);
  // Reconstruct solution data from contiguous workspace
  std::copy(solution_workspace_.begin(), solution_workspace_.end(),
            solution->begin());
  return {0, 0};
}

/// \cond
template <typename LinearSolverRegistrars>
// NOLINTNEXTLINE
PUP::able::PUP_ID ExplicitInverse<LinearSolverRegistrars>::my_PUP_ID = 0;
/// \endcond

}  // namespace LinearSolver::Serial
