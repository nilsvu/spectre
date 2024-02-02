// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <type_traits>
#include <utility>

#include "DataStructures/DynamicMatrix.hpp"
#include "DataStructures/DynamicVector.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver {
namespace gmres::detail {

// Perform an Arnoldi orthogonalization to find a new `operand` that is
// orthogonal to all vectors in `basis_history`. Appends a new column to the
// `orthogonalization_history` that holds the inner product of the intermediate
// `operand` with each vector in the `basis_history` and itself.
template <typename VarsType>
void arnoldi_orthogonalize(const gsl::not_null<VarsType*> operand,
                           const gsl::not_null<blaze::DynamicMatrix<double>*>
                               orthogonalization_history,
                           const std::vector<VarsType>& basis_history,
                           const size_t iteration) {
  // Resize matrix and make sure the new entries that are not being filled below
  // are zero.
  orthogonalization_history->resize(iteration + 2, iteration + 1);
  for (size_t j = 0; j < iteration; ++j) {
    (*orthogonalization_history)(iteration + 1, j) = 0.;
  }
  // Arnoldi orthogonalization
  for (size_t j = 0; j < iteration + 1; ++j) {
    const double orthogonalization = inner_product(basis_history[j], *operand);
    (*orthogonalization_history)(j, iteration) = orthogonalization;
    *operand -= orthogonalization * basis_history[j];
  }
  (*orthogonalization_history)(iteration + 1, iteration) =
      sqrt(inner_product(*operand, *operand));
  // Avoid an FPE if the new operand norm is exactly zero. In that case the
  // problem is solved and the algorithm will terminate (see Proposition 9.3 in
  // \cite Saad2003). Since there will be no next iteration we don't need to
  // normalize the operand.
  if (UNLIKELY((*orthogonalization_history)(iteration + 1, iteration) == 0.)) {
    return;
  }
  *operand /= (*orthogonalization_history)(iteration + 1, iteration);
}

// Solve the linear least-squares problem `||beta - H * y||` for `y`, where `H`
// is the Hessenberg matrix given by `orthogonalization_history` and `beta` is
// the vector `(initial_residual, 0, 0, ...)` by updating the QR decomposition
// of `H` from the previous iteration with a Givens rotation.
void solve_minimal_residual(
    gsl::not_null<blaze::DynamicMatrix<double>*> orthogonalization_history,
    gsl::not_null<blaze::DynamicVector<double>*> residual_history,
    gsl::not_null<blaze::DynamicVector<double>*> givens_sine_history,
    gsl::not_null<blaze::DynamicVector<double>*> givens_cosine_history,
    size_t iteration);

// Find the vector that minimizes the residual by inverting the upper
// triangular matrix obtained above.
blaze::DynamicVector<double> minimal_residual_vector(
    const blaze::DynamicMatrix<double>& orthogonalization_history,
    const blaze::DynamicVector<double>& residual_history);

}  // namespace gmres::detail

namespace Serial {

/// Disables the iteration callback at compile-time
struct NoIterationCallback {};

/// \cond
template <typename VarsType, typename Preconditioner,
          typename LinearSolverRegistrars>
struct Gmres;
/// \endcond

namespace Registrars {

/// Registers the `LinearSolver::Serial::Gmres` linear solver.
template <typename VarsType>
struct Gmres {
  template <typename LinearSolverRegistrars>
  using f = Serial::Gmres<VarsType, LinearSolver<LinearSolverRegistrars>,
                          LinearSolverRegistrars>;
};
}  // namespace Registrars

/*!
 * \brief A serial GMRES iterative solver for nonsymmetric linear systems of
 * equations.
 *
 * This is an iterative algorithm to solve general linear equations \f$Ax=b\f$
 * where \f$A\f$ is a linear operator. See \cite Saad2003, chapter 6.5 for a
 * description of the GMRES algorithm and Algorithm 9.6 for this implementation.
 * It is matrix-free, which means the operator \f$A\f$ needs not be provided
 * explicity as a matrix but only the operator action \f$A(x)\f$ must be
 * provided for an argument \f$x\f$.
 *
 * The GMRES algorithm does not require the operator \f$A\f$ to be symmetric or
 * positive-definite. Note that other algorithms such as conjugate gradients may
 * be more efficient for symmetric positive-definite operators.
 *
 * \par Convergence:
 * Given a set of \f$N_A\f$ equations (e.g. through an \f$N_A\times N_A\f$
 * matrix) the GMRES algorithm will converge to numerical precision in at most
 * \f$N_A\f$ iterations. However, depending on the properties of the linear
 * operator, an approximate solution can ideally be obtained in only a few
 * iterations. See \cite Saad2003, section 6.11.4 for details on the convergence
 * of the GMRES algorithm.
 *
 * \par Restarting:
 * This implementation of the GMRES algorithm supports restarting, as detailed
 * in \cite Saad2003, section 6.5.5. Since the GMRES algorithm iteratively
 * builds up an orthogonal basis of the solution space the cost of each
 * iteration increases linearly with the number of iterations. Therefore it is
 * sometimes helpful to restart the algorithm every \f$N_\mathrm{restart}\f$
 * iterations, discarding the set of basis vectors and starting again from the
 * current solution estimate. This strategy can improve the performance of the
 * solver, but note that the solver can stagnate for non-positive-definite
 * operators and is not guaranteed to converge within \f$N_A\f$ iterations
 * anymore. Set the `restart` argument of the constructor to
 * \f$N_\mathrm{restart}\f$ to activate restarting, or set it to 'None' to
 * deactivate restarting.
 *
 * \par Preconditioning:
 * This implementation of the GMRES algorithm also supports preconditioning.
 * You can provide a linear operator \f$P\f$ that approximates the inverse of
 * the operator \f$A\f$ to accelerate the convergence of the linear solve.
 * The algorithm is right-preconditioned, which allows the preconditioner to
 * change in every iteration ("flexible" variant). See \cite Saad2003, sections
 * 9.3.2 and 9.4.1 for details. This implementation follows Algorithm 9.6 in
 * \cite Saad2003.
 *
 * \par Improvements:
 * Further improvements can potentially be implemented for this algorithm, see
 * e.g. \cite Ayachour2003.
 *
 * \example
 * \snippet NumericalAlgorithms/LinearSolver/Test_Gmres.cpp gmres_example
 */
template <typename VarsType, typename Preconditioner = NoPreconditioner,
          typename LinearSolverRegistrars =
              tmpl::list<Registrars::Gmres<VarsType>>>
class Gmres final : public PreconditionedLinearSolver<Preconditioner,
                                                      LinearSolverRegistrars> {
 private:
  using Base =
      PreconditionedLinearSolver<Preconditioner, LinearSolverRegistrars>;

  struct ConvergenceCriteria {
    using type = Convergence::Criteria;
    static constexpr Options::String help =
        "Determine convergence of the algorithm";
  };
  struct Restart {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "Iterations to run before restarting, or 'None' to disable restarting. "
        "Note that the solver is not guaranteed to converge anymore if you "
        "enable restarting.";
    static type suggested_value() { return {}; }
  };
  struct Verbosity {
    using type = ::Verbosity;
    static constexpr Options::String help = "Logging verbosity";
  };

 public:
  static constexpr Options::String help =
      "A serial GMRES iterative solver for nonsymmetric linear systems of\n"
      "equations Ax=b. It will converge to numerical precision in at most N_A\n"
      "iterations, where N_A is the number of equations represented by the\n"
      "linear operator A, but will ideally converge to a reasonable\n"
      "approximation of the solution x in only a few iterations.\n"
      "\n"
      "Preconditioning: Specify a preconditioner to run in every GMRES "
      "iteration to accelerate the solve, or 'None' to disable "
      "preconditioning. The choice of preconditioner can be crucial to obtain "
      "good convergence.\n"
      "\n"
      "Restarting: It is sometimes helpful to restart the algorithm every\n"
      "N_restart iterations to speed it up. Note that it can stagnate for\n"
      "non-positive-definite matrices and is not guaranteed to converge\n"
      "within N_A iterations anymore when restarting is activated.\n"
      "Activate restarting by setting the 'Restart' option to N_restart, or\n"
      "deactivate restarting by setting it to 'None'.";
  using options = tmpl::flatten<tmpl::list<
      ConvergenceCriteria, Verbosity, Restart,
      tmpl::conditional_t<std::is_same_v<Preconditioner, NoPreconditioner>,
                          tmpl::list<>, typename Base::PreconditionerOption>>>;

  Gmres(Convergence::Criteria convergence_criteria, ::Verbosity verbosity,
        std::optional<size_t> restart = std::nullopt,
        std::optional<typename Base::PreconditionerType> local_preconditioner =
            std::nullopt,
        const Options::Context& context = {});

  Gmres() = default;
  Gmres(Gmres&&) = default;
  Gmres& operator=(Gmres&&) = default;
  ~Gmres() override = default;

  Gmres(const Gmres& rhs);
  Gmres& operator=(const Gmres& rhs);

  std::unique_ptr<LinearSolver<LinearSolverRegistrars>> get_clone()
      const override {
    return std::make_unique<Gmres>(*this);
  }

  /// \cond
  explicit Gmres(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Gmres);  // NOLINT
  /// \endcond

  void initialize() {
    orthogonalization_history_.reserve(restart_ + 1);
    residual_history_.reserve(restart_ + 1);
    givens_sine_history_.reserve(restart_);
    givens_cosine_history_.reserve(restart_);
    basis_history_.resize(restart_ + 1);
    preconditioned_basis_history_.resize(restart_);
  }

  const Convergence::Criteria& convergence_criteria() const {
    return convergence_criteria_;
  }
  ::Verbosity verbosity() const { return verbosity_; }
  size_t restart() const { return restart_; }

  void pup(PUP::er& p) override {  // NOLINT
    Base::pup(p);
    p | convergence_criteria_;
    p | verbosity_;
    p | restart_;
    if (p.isUnpacking()) {
      initialize();
    }
  }

  template <typename LinearOperator, typename SourceType,
            typename... OperatorArgs,
            typename IterationCallback = NoIterationCallback>
  Convergence::HasConverged solve(
      gsl::not_null<VarsType*> initial_guess_in_solution_out,
      const LinearOperator& linear_operator, const SourceType& source,
      const std::tuple<OperatorArgs...>& operator_args = std::tuple{},
      const IterationCallback& iteration_callback =
          NoIterationCallback{}) const;

  void reset() override {
    basis_history_.clear();
    basis_history_.resize(restart_ + 1);
    preconditioned_basis_history_.clear();
    preconditioned_basis_history_.resize(restart_);
    Base::reset();
  }

 private:
  Convergence::Criteria convergence_criteria_{};
  ::Verbosity verbosity_{::Verbosity::Verbose};
  size_t restart_{};

  // Memory buffers to avoid re-allocating memory for successive solves:
  // The `orthogonalization_history_` is built iteratively from inner products
  // between existing and potential basis vectors and then Givens-rotated to
  // become upper-triangular.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable blaze::DynamicMatrix<double> orthogonalization_history_{};
  // The `residual_history_` holds the remaining residual in its last entry, and
  // the other entries `g` "source" the minimum residual vector `y` in
  // `R * y = g` where `R` is the upper-triangular `orthogonalization_history_`.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable blaze::DynamicVector<double> residual_history_{};
  // These represent the accumulated Givens rotations up to the current
  // iteration.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable blaze::DynamicVector<double> givens_sine_history_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable blaze::DynamicVector<double> givens_cosine_history_{};
  // These represent the orthogonal Krylov-subspace basis that is constructed
  // iteratively by Arnoldi-orthogonalizing a new vector in each iteration and
  // appending it to the `basis_history_`.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::vector<VarsType> basis_history_{};
  // When a preconditioner is used it is applied to each new basis vector. The
  // preconditioned basis is used to construct the solution when the algorithm
  // has converged.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::vector<VarsType> preconditioned_basis_history_{};
};

template <typename VarsType, typename Preconditioner,
          typename LinearSolverRegistrars>
Gmres<VarsType, Preconditioner, LinearSolverRegistrars>::Gmres(
    Convergence::Criteria convergence_criteria, ::Verbosity verbosity,
    std::optional<size_t> restart,
    std::optional<typename Base::PreconditionerType> local_preconditioner,
    const Options::Context& context)
    // clang-tidy: trivially copyable
    : Base(std::move(local_preconditioner)),
      convergence_criteria_(std::move(convergence_criteria)),  // NOLINT
      verbosity_(std::move(verbosity)),                        // NOLINT
      restart_(restart.value_or(convergence_criteria_.max_iterations)) {
  if (restart_ == 0) {
    PARSE_ERROR(context,
                "Can't restart every '0' iterations. Set to a nonzero "
                "number, or to 'None' if you meant to disable restarting.");
  }
  initialize();
}

// Define copy constructors. They don't have to copy the memory buffers but
// only resize them. They take care of copying the preconditioner by calling
// into the base class.
template <typename VarsType, typename Preconditioner,
          typename LinearSolverRegistrars>
Gmres<VarsType, Preconditioner, LinearSolverRegistrars>::Gmres(const Gmres& rhs)
    : Base(rhs),
      convergence_criteria_(rhs.convergence_criteria_),
      verbosity_(rhs.verbosity_),
      restart_(rhs.restart_) {
  initialize();
}
template <typename VarsType, typename Preconditioner,
          typename LinearSolverRegistrars>
Gmres<VarsType, Preconditioner, LinearSolverRegistrars>&
Gmres<VarsType, Preconditioner, LinearSolverRegistrars>::operator=(
    const Gmres& rhs) {
  Base::operator=(rhs);
  convergence_criteria_ = rhs.convergence_criteria_;
  verbosity_ = rhs.verbosity_;
  restart_ = rhs.restart_;
  initialize();
  return *this;
}

/// \cond
template <typename VarsType, typename Preconditioner,
          typename LinearSolverRegistrars>
Gmres<VarsType, Preconditioner, LinearSolverRegistrars>::Gmres(
    CkMigrateMessage* m)
    : Base(m) {}
/// \endcond

template <typename VarsType, typename Preconditioner,
          typename LinearSolverRegistrars>
template <typename LinearOperator, typename SourceType,
          typename... OperatorArgs, typename IterationCallback>
Convergence::HasConverged
Gmres<VarsType, Preconditioner, LinearSolverRegistrars>::solve(
    const gsl::not_null<VarsType*> initial_guess_in_solution_out,
    const LinearOperator& linear_operator, const SourceType& source,
    const std::tuple<OperatorArgs...>& operator_args,
    const IterationCallback& iteration_callback) const {
  constexpr bool use_preconditioner =
      not std::is_same_v<Preconditioner, NoPreconditioner>;
  constexpr bool use_iteration_callback =
      not std::is_same_v<IterationCallback, NoIterationCallback>;

  // Could pre-allocate memory for the basis-history vectors here. Not doing
  // that for now because we don't know how many iterations we'll need.
  // Estimating the number of iterations and pre-allocating memory is a possible
  // performance optimization.

  auto& solution = *initial_guess_in_solution_out;
  Convergence::HasConverged has_converged{};
  size_t iteration = 0;

  while (not has_converged) {
    const auto& initial_guess = *initial_guess_in_solution_out;
    auto& initial_operand = basis_history_[0];
    // Apply the linear operator to the initial guess. This can be skipped if
    // the initial guess is zero, because then the linear operator applied to it
    // is also zero.
    if (equal_within_roundoff(initial_guess, 0.)) {
      initial_operand = source;
    } else {
      std::apply(
          linear_operator,
          std::tuple_cat(std::forward_as_tuple(make_not_null(&initial_operand),
                                               initial_guess),
                         operator_args));
      initial_operand *= -1.;
      initial_operand += source;
    }
    const double initial_residual_magnitude =
        sqrt(inner_product(initial_operand, initial_operand));
    has_converged = Convergence::HasConverged{convergence_criteria_, iteration,
                                              initial_residual_magnitude,
                                              initial_residual_magnitude};
    if constexpr (use_iteration_callback) {
      iteration_callback(has_converged);
    }
    if (UNLIKELY(has_converged)) {
      break;
    }
    initial_operand /= initial_residual_magnitude;
    residual_history_.resize(1);
    residual_history_[0] = initial_residual_magnitude;
    for (size_t k = 0; k < restart_; ++k) {
      auto& operand = basis_history_[k + 1];
      if constexpr (use_preconditioner) {
        if (this->has_preconditioner()) {
          // Begin the preconditioner at an initial guess of 0. Not all
          // preconditioners take the initial guess into account.
          preconditioned_basis_history_[k] =
              make_with_value<VarsType>(initial_operand, 0.);
          this->preconditioner().solve(
              make_not_null(&preconditioned_basis_history_[k]), linear_operator,
              basis_history_[k], operator_args);
        }
      }
      std::apply(linear_operator,
                 std::tuple_cat(std::forward_as_tuple(
                                    make_not_null(&operand),
                                    this->has_preconditioner()
                                        ? preconditioned_basis_history_[k]
                                        : basis_history_[k]),
                                operator_args));
      // Find a new orthogonal basis vector of the Krylov subspace
      gmres::detail::arnoldi_orthogonalize(
          make_not_null(&operand), make_not_null(&orthogonalization_history_),
          basis_history_, k);
      // Least-squares solve for the minimal residual
      gmres::detail::solve_minimal_residual(
          make_not_null(&orthogonalization_history_),
          make_not_null(&residual_history_),
          make_not_null(&givens_sine_history_),
          make_not_null(&givens_cosine_history_), k);
      ++iteration;
      has_converged = Convergence::HasConverged{
          convergence_criteria_, iteration, abs(residual_history_[k + 1]),
          initial_residual_magnitude};
      if constexpr (use_iteration_callback) {
        iteration_callback(has_converged);
      }
      if (UNLIKELY(has_converged)) {
        break;
      }
    }
    // Find the vector w.r.t. the constructed orthogonal basis of the Krylov
    // subspace that minimizes the residual
    const auto minres = gmres::detail::minimal_residual_vector(
        orthogonalization_history_, residual_history_);
    // Construct the solution from the orthogonal basis and the minimal residual
    // vector
    for (size_t i = 0; i < minres.size(); ++i) {
      solution += minres[i] * gsl::at(this->has_preconditioner()
                                          ? preconditioned_basis_history_
                                          : basis_history_,
                                      i);
    }
  }
  return has_converged;
}

/// \cond
template <typename VarsType, typename Preconditioner,
          typename LinearSolverRegistrars>
// NOLINTNEXTLINE
PUP::able::PUP_ID
    Gmres<VarsType, Preconditioner, LinearSolverRegistrars>::my_PUP_ID = 0;
/// \endcond

}  // namespace Serial
}  // namespace LinearSolver
