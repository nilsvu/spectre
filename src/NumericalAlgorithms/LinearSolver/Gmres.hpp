// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/Submatrix.h>
#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Utilities/Gsl.hpp"

#include "Parallel/Printf.hpp"

namespace LinearSolver {
namespace gmres_detail {
std::pair<DenseVector<double>, double> minimal_residual(
    const DenseMatrix<double>& orthogonalization_history,
    const double initial_residual_magnitude) noexcept;
}  // namespace gmres_detail

namespace Serial {

template <typename VarsType>
struct IdentityPreconditioner {
  VarsType operator()(const VarsType& arg) const noexcept { return arg; }
};

template <typename VarsType>
class Gmres {
 private:
  struct ConvergenceCriteria {
    using type = Convergence::Criteria;
    static constexpr OptionString help =
        "Determine convergence of the algorithm";
  };
  struct Restart {
    using type = size_t;
    static constexpr OptionString help = "Iterations to run before restarting";
    static size_t default_value() noexcept { return 0; }
  };
  struct Verbosity {
    using type = ::Verbosity;
    static constexpr OptionString help = "Logging verbosity";
  };

 public:
  static constexpr OptionString help = "A GMRES linear solver";
  using options = tmpl::list<ConvergenceCriteria, Verbosity, Restart>;

  Gmres(Convergence::Criteria convergence_criteria, ::Verbosity verbosity,
        size_t restart = 0) noexcept
      : convergence_criteria_(std::move(convergence_criteria)),
        verbosity_(std::move(verbosity)),
        restart_(restart > 0 ? restart : convergence_criteria_.max_iterations) {
    initialize();
  }

  Gmres() = default;
  Gmres(const Gmres& /*rhs*/) = default;
  Gmres& operator=(const Gmres& /*rhs*/) = default;
  Gmres(Gmres&& /*rhs*/) noexcept = default;
  Gmres& operator=(Gmres&& /*rhs*/) noexcept = default;
  ~Gmres() = default;

  void initialize() noexcept {
    orthogonalization_history_.reserve(restart_ + 1);
    basis_history_.resize(restart_);
    preconditioned_basis_history_.resize(restart_);
  }

  Convergence::Reason convergence_reason() const noexcept {
    ASSERT(convergence_reason_,
           "Tried to retrieve the convergence reason, but has not performed a "
           "solve yet.");
    return *convergence_reason_;
  }

  void pup(PUP::er& p) noexcept {
    p | convergence_criteria_;
    p | verbosity_;
    p | restart_;
    if (p.isUnpacking()) {
      initialize();
    }
  }

  template <typename LinearOperator, typename SourceType,
            typename Preconditioner = IdentityPreconditioner<VarsType>>
  VarsType operator()(LinearOperator&& linear_operator,
                      const SourceType& source, const VarsType& initial_guess,
                      const Preconditioner& preconditioner =
                          IdentityPreconditioner<VarsType>{}) const noexcept {
    constexpr bool use_preconditioner =
        not cpp17::is_same_v<Preconditioner, IdentityPreconditioner<VarsType>>;

    // Build matrix for testing
    // Only works if there are no internal boundaries
    // const size_t num_points =
    //     initial_guess.element_data.number_of_grid_points();
    // const size_t size = initial_guess.element_data.size();
    // DenseMatrix<double> matrix{size, size};
    // for (size_t i = 0; i < size; i++) {
    //   VarsType unit_vector{num_points};
    //   unit_vector.element_data = typename VarsType::Vars{num_points, 0.};
    //   unit_vector.element_data.data()[i] = 1.;
    //   const auto col = linear_operator(unit_vector);
    //   for (size_t j = 0; j < size; j++) {
    //     matrix(i, j) = col.element_data.data()[j];
    //   }
    // }
    Parallel::printf("\n\n--- Serial GMRES --- \n\n");
    // Parallel::printf("Solving matrix :\n % s\n ", matrix);
    Parallel::printf("Source: %s\n", source.element_data);
    Parallel::printf("initial_guess: %s\n", initial_guess.element_data);

    auto result = initial_guess;
    convergence_reason_ = boost::none;
    size_t iteration = 0;
    // Can allocated memory for the operand_ be kept around for successive
    // solves?
    VarsType operand_{};
    while (not convergence_reason_) {
      operand_ = source - linear_operator(result);
      // Parallel::printf("r: %s\n", operand_.element_data);
      // Parallel::printf("r.boundary: %d\n", operand_.boundary_data.size());
      const double initial_residual_magnitude =
          sqrt(inner_product(operand_, operand_));
      Parallel::printf("Init residual: %e\n", initial_residual_magnitude);
      convergence_reason_ = Convergence::criteria_match(
          convergence_criteria_, iteration, initial_residual_magnitude,
          initial_residual_magnitude);
      if (convergence_reason_) {
        break;
      }
      operand_ /= initial_residual_magnitude;
      // Parallel::printf("  init q: %s\n", operand_);
      basis_history_[0] = operand_;

      std::pair<DenseVector<double>, double> minres_and_magnitude{};
      for (size_t k = 0; k < restart_; k++) {
        Parallel::printf("Iteration %zu:\n", k);
        if (use_preconditioner) {
          preconditioned_basis_history_[k] = preconditioner(operand_);
          // Parallel::printf("  z: %s\n", preconditioned_basis_history_[k]);
        }
        operand_ = linear_operator(
            use_preconditioner ? preconditioned_basis_history_[k] : operand_);
        // Parallel::printf("  q: %s\n", operand_);
        // Resize matrix
        orthogonalization_history_.resize(k + 2, k + 1);
        // Make sure the new entries are zero
        for (size_t i = 0; i < orthogonalization_history_.rows(); i++) {
          orthogonalization_history_(
              i, orthogonalization_history_.columns() - 1) = 0.;
        }
        for (size_t j = 0; j < orthogonalization_history_.columns(); j++) {
          orthogonalization_history_(orthogonalization_history_.rows() - 1, j) =
              0.;
        }
        // Arnoldi orthogonalization
        // Parallel::printf("  resized_H: %s\n", orthogonalization_history_);
        for (size_t j = 0; j < k + 1; j++) {
          const double orthogonalization =
              inner_product(basis_history_[j], operand_);
          orthogonalization_history_(j, k) = orthogonalization;
          operand_ -= orthogonalization * basis_history_[j];
        }
        orthogonalization_history_(k + 1, k) =
            sqrt(inner_product(operand_, operand_));
        // Parallel::printf("  orthogonalization_history_:\n%s\n",
        //                  orthogonalization_history_);
        // Least-squares solve for the minimal residual
        minres_and_magnitude = gmres_detail::minimal_residual(
            orthogonalization_history_, initial_residual_magnitude);
        // Parallel::printf("  minres: %s\n", minres_and_magnitude.first);
        Parallel::printf("  res: %e\n", minres_and_magnitude.second);
        convergence_reason_ = Convergence::criteria_match(
            convergence_criteria_, iteration + 1, minres_and_magnitude.second,
            initial_residual_magnitude);
        // Parallel::printf("  conv: %s\n", convergence);
        if (UNLIKELY(convergence_reason_)) {
          break;
        } else if (k + 1 < restart_) {
          // Parallel::printf("  norm for next: %e\n",
          //                  orthogonalization_history_(k + 1, k));
          operand_ /= orthogonalization_history_(k + 1, k);
          basis_history_[k + 1] = operand_;
        }
        iteration++;
      }
      for (size_t i = 0; i < minres_and_magnitude.first.size(); i++) {
        result += minres_and_magnitude.first[i] *
                  gsl::at(use_preconditioner ? preconditioned_basis_history_
                                             : basis_history_,
                          i);
      }
    }
    Parallel::printf("\n=> GMRES is done. Result:\n%s\n", result.element_data);
    return result;
  };

 private:
  Convergence::Criteria convergence_criteria_{};
  ::Verbosity verbosity_{::Verbosity::Verbose};
  size_t restart_{};

  mutable DenseMatrix<double> orthogonalization_history_{};
  // VarsType operand_{};
  mutable std::vector<VarsType> basis_history_{};
  mutable std::vector<VarsType> preconditioned_basis_history_{};

  mutable boost::optional<Convergence::Reason> convergence_reason_{};
};

}  // namespace Serial
}  // namespace LinearSolver
