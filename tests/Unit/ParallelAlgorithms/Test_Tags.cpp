// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct TestLabel {};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Tags",
                  "[Unit][ParallelAlgorithms]") {
  TestHelpers::db::test_simple_tag<Parallel::Tags::IterationId<TestLabel>>(
      "IterationId(TestLabel)");
}
