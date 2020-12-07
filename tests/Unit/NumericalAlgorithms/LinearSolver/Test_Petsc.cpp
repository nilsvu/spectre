// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/LinearSolver/TestHelpers.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/Petsc.hpp"
#include "Utilities/Gsl.hpp"

namespace helpers = TestHelpers::LinearSolver;

namespace LinearSolver::Serial {

namespace {
struct ScalarField : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <typename Tag>
struct SomePrefix : db::PrefixTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.LinearSolver.Serial.Petsc",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  {
    INFO("Solve a symmetric 2x2 matrix");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 1.;
    matrix(1, 1) = 3.;
    const helpers::ApplyMatrix linear_operator{std::move(matrix)};
    const DenseVector<double> source{1., 2.};
    DenseVector<double> initial_guess_in_solution_out{2., 1.};
    const DenseVector<double> expected_solution{0.0909090909090909,
                                                0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Petsc<> petsc{convergence_criteria};
    const auto has_converged = petsc.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK(has_converged.residual_magnitude() <= 1.e-14);
    CHECK(has_converged.initial_residual_magnitude() ==
          approx(8.54400374531753));
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
  }
}

}  // namespace LinearSolver::Serial
