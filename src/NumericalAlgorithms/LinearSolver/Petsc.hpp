// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <petscksp.h>
#include <pup.h>
#include <type_traits>

#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"

extern PetscErrorCode usermult(Mat m, Vec x, Vec y) {
  PetscErrorCode ierr = 0;
  MatShellGetContext(m, (void**)&myctx);
  ierr = MatMult(m, x, y);
  printf("Call\n");
  return ierr;
}

namespace LinearSolver::Serial {

/// \cond
template <typename LinearSolverRegistrars>
struct Petsc;
/// \endcond

namespace Registrars {
struct Petsc {
  template <typename LinearSolverRegistrars>
  using f = Serial::Petsc<LinearSolverRegistrars>;
};
}  // namespace Registrars

template <typename LinearSolverRegistrars = tmpl::list<Registrars::Petsc>>
class Petsc : public LinearSolver<LinearSolverRegistrars>::Inherit {
 private:
  struct ConvergenceCriteria {
    using type = Convergence::Criteria;
    static constexpr Options::String help =
        "Determine convergence of the algorithm";
  };

 public:
  static constexpr Options::String help = "Petsc";
  using options = tmpl::list<ConvergenceCriteria>;

  explicit Petsc(Convergence::Criteria convergence_criteria)
      : convergence_criteria_(std::move(convergence_criteria)) {}

  Petsc() = default;
  Petsc(const Petsc& /*rhs*/) = default;
  Petsc& operator=(const Petsc& /*rhs*/) = default;
  Petsc(Petsc&& /*rhs*/) = default;
  Petsc& operator=(Petsc&& /*rhs*/) = default;
  ~Petsc() = default;

  /// \cond
  explicit Petsc(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Petsc);  // NOLINT
  /// \endcond

  const Convergence::Criteria& convergence_criteria() const noexcept {
    return convergence_criteria_;
  }

  void pup(PUP::er& p) noexcept override {  // NOLINT
    p | convergence_criteria_;
  }

  void reset() noexcept override {}

  struct Ctx {};

  template <typename LinearOperator, typename VarsType, typename SourceType>
  Convergence::HasConverged solve(
      gsl::not_null<VarsType*> initial_guess_in_solution_out,
      LinearOperator&& linear_operator,
      const SourceType& source) const noexcept {
    int argc = 0;
    char** argv;
    char help[] = "halp.\n\n";
    PetscInitialize(&argc, &argv, NULL, help);

    const size_t size = source.size();
    Ctx ctx;
    Mat s;
    auto ierr = MatCreateShell(PETSC_COMM_WORLD, size, size, PETSC_DECIDE,
                          PETSC_DECIDE, &ctx, &s);
    ierr = MatShellSetOperation(s, MATOP_MULT, (void (*)(void))usermult);

    KSP solver;
    KSPCreate(PETSC_COMM_WORLD, &solver);
    KSPSetFromOptions(solver);
    KSPSetOperators(solver, s, s);

    return {0, 0};

    // KSP ksp;
    // Mat A;
    // Vec x, b;
    // int n, its;
    // MatCreateMFFD(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, &A);
    // MatSetFromOptions(A);
    // /* (code to assemble matrix not shown) */
    // VecCreate(PETSC_COMM_WORLD, &x);
    // VecSetSizes(x, PETSC_DECIDE, n);
    // VecSetFromOptions(x);
    // VecDuplicate(x, &b);
    // /* (code to assemble RHS vector not shown)*/
    // KSPCreate(PETSC_COMM_WORLD, &ksp);
    // KSPSetOperators(ksp, A, A, DIFFERENT_NONZERO_PATTERN);
    // KSPSetFromOptions(ksp);
    // KSPSolve(ksp, b, x, &its);
    // KSPDestroy(ksp);
  }

 private:
  Convergence::Criteria convergence_criteria_{};
};

/// \cond
template <typename LinearSolverRegistrars>
// NOLINTNEXTLINE
PUP::able::PUP_ID Petsc<LinearSolverRegistrars>::my_PUP_ID = 0;
/// \endcond

}  // namespace LinearSolver::Serial
