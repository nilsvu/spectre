// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <blaze/math/IdentityMatrix.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/LinearSolver/ExplicitInverse.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarFieldTag {
  using type = Scalar<DataVector>;
};
}  // namespace

namespace LinearSolver::Serial {

SPECTRE_TEST_CASE("Unit.LinearSolver.Serial.ExplicitInverse",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  using SubdomainData = ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
      1, tmpl::list<ScalarFieldTag>>;
  SubdomainData result_buffer{3};
  result_buffer.overlap_data.emplace(
      ::LinearSolver::Schwarz::OverlapId<1>{Direction<1>::lower_xi(),
                                            ElementId<1>{0}},
      typename SubdomainData::OverlapData{2});
  ExplicitInverse solver{};
  solver.prepare(
      [](const gsl::not_null<SubdomainData*> result,
         const SubdomainData& source) noexcept { return *result = source; },
      make_not_null(&result_buffer));
  CHECK(solver.size() == 5);
  CHECK_MATRIX_APPROX(solver.matrix_representation(),
                      blaze::IdentityMatrix<double>(5));
}

}  // namespace LinearSolver::Serial
