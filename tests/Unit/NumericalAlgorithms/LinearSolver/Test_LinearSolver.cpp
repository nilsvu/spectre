// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DenseVector.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"

namespace LinearSolver::Serial {

namespace {

struct Identity {
  void operator()(const gsl::not_null<DenseVector<double>*> result,
                  const DenseVector<double>& operand) const {
    *result = operand;
  }
};

template <typename LinearSolverType>
void test_linear_solve(const std::unique_ptr<LinearSolverType>& linear_solver) {
  const Identity linear_operator{};
  DenseVector<double> buffer(2);

  linear_solver->prepare(linear_operator, make_not_null(&buffer));
  CHECK(linear_solver->is_ready());

  const DenseVector<double> source{1., 2.};
  std::fill(buffer.begin(), buffer.end(), 0.);
  const auto has_converged =
      linear_solver->solve(make_not_null(&buffer), linear_operator, source);
  CHECK(has_converged);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.LinearSolver.Serial.LinearSolver",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  {
    INFO("Factory-creation");
    using LinearSolverFactory =
        LinearSolver<tmpl::list<Registrars::Gmres<DenseVector<double>>>>;
    {
      INFO("Unpreconditioned GMRES");
      auto linear_solver =
          TestHelpers::test_factory_creation<LinearSolverFactory>(
              "Gmres:\n"
              "  ConvergenceCriteria:\n"
              "    MaxIterations: 2\n"
              "    RelativeResidual: 0.5\n"
              "    AbsoluteResidual: 0.1\n"
              "  Restart: 10\n"
              "  Verbosity: Verbose\n"
              "  Preconditioner: None");
      test_linear_solve(linear_solver);
    }
    {
      INFO("Nested GMRES");
      const auto linear_solver =
          TestHelpers::test_factory_creation<LinearSolverFactory>(
              "Gmres:\n"
              "  ConvergenceCriteria:\n"
              "    MaxIterations: 5\n"
              "    RelativeResidual: 0.5\n"
              "    AbsoluteResidual: 0.1\n"
              "  Restart: None\n"
              "  Verbosity: Verbose\n"
              "  Preconditioner:\n"
              "    Gmres:\n"
              "      ConvergenceCriteria:\n"
              "        MaxIterations: 2\n"
              "        RelativeResidual: 0.9\n"
              "        AbsoluteResidual: 0.2\n"
              "      Restart: None\n"
              "      Verbosity: Silent\n"
              "      Preconditioner: None\n");
      test_linear_solve(linear_solver);
    }
  }
}

}  // namespace LinearSolver::Serial
