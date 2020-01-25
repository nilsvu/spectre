// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace LinearSolver {
namespace Serial {

namespace {
struct ScalarField : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <typename Tag>
struct SomePrefix : db::PrefixTag {
  using type = tmpl::type_from<Tag>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.LinearSolver.Serial.Gmres",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  {
    INFO("Solve a symmetric 2x2 matrix");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 1.;
    matrix(1, 1) = 3.;
    const DenseVector<double> source{1., 2.};
    const DenseVector<double> initial_guess{2., 1.};
    const DenseVector<double> expected_solution{0.0909090909090909,
                                                0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    Gmres<DenseVector<double>> gmres{convergence_criteria,
                                     ::Verbosity::Verbose};
    const auto linear_operator =
        [&matrix](const DenseVector<double>& arg) noexcept {
          return matrix * arg;
        };
    auto solution = gmres(linear_operator, source, initial_guess);
    CHECK(gmres.convergence_reason() == Convergence::Reason::AbsoluteResidual);
    CHECK_ITERABLE_APPROX(solution, expected_solution);
    {
      INFO("Check that a solved system terminates early");
      auto another_solution = gmres(linear_operator, source, solution);
      CHECK(gmres.convergence_reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK_ITERABLE_APPROX(solution, another_solution);
    }
  }
  {
    INFO("Solve a non-symmetric 2x2 matrix");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 3.;
    matrix(1, 1) = 1.;
    const DenseVector<double> source{1., 2.};
    const DenseVector<double> initial_guess{2., 1.};
    const DenseVector<double> expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    Gmres<DenseVector<double>> gmres{convergence_criteria,
                                     ::Verbosity::Verbose};
    auto solution = gmres(
        [&matrix](const DenseVector<double>& arg) noexcept {
          return matrix * arg;
        },
        source, initial_guess);
    CHECK(gmres.convergence_reason() == Convergence::Reason::AbsoluteResidual);
    CHECK_ITERABLE_APPROX(solution, expected_solution);
  }
  {
    INFO("Solve a matrix-free linear operator");
    const auto linear_operator = [](const DataVector& arg) noexcept {
      return DataVector{arg[0] * 4. + arg[1], arg[0] * 3. + arg[1]};
    };
    const DataVector source{1., 2.};
    const DataVector initial_guess{2., 1.};
    const DataVector expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    Gmres<DataVector> gmres{convergence_criteria, ::Verbosity::Verbose};
    auto solution = gmres(linear_operator, source, initial_guess);
    CHECK(gmres.convergence_reason() == Convergence::Reason::AbsoluteResidual);
    CHECK_ITERABLE_APPROX(solution, expected_solution);
  }
  {
    INFO("Solve a matrix-free linear operator with Variables");
    using Vars = Variables<tmpl::list<ScalarField>>;
    constexpr size_t num_points = 2;
    const auto linear_operator = [](const Vars& arg) noexcept {
      auto& data = get(get<ScalarField>(arg));
      Vars result{num_points};
      get(get<ScalarField>(result)) =
          DataVector{data[0] * 4. + data[1], data[0] * 3. + data[1]};
      return result;
    };
    // Adding a prefix to make sure prefixed sources work as well
    Variables<tmpl::list<SomePrefix<ScalarField>>> source{num_points};
    get(get<SomePrefix<ScalarField>>(source)) = DataVector{1., 2.};
    Vars initial_guess{num_points};
    get(get<ScalarField>(initial_guess)) = DataVector{2., 1.};
    const DataVector expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    Gmres<Vars> gmres{convergence_criteria, ::Verbosity::Verbose};
    auto solution =
        gmres(linear_operator, std::move(source), std::move(initial_guess));
    CHECK(gmres.convergence_reason() == Convergence::Reason::AbsoluteResidual);
    CHECK_ITERABLE_APPROX(get(get<ScalarField>(solution)), expected_solution);
  }
}

}  // namespace Serial
}  // namespace LinearSolver
