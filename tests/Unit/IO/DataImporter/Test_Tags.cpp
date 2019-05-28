// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "IO/DataImporter/Tags.hpp"

SPECTRE_TEST_CASE("Unit.IO.DataImporter.Tags", "[Unit][IO]") {
  CHECK(importer::Tags::RegisteredElements::name() == "RegisteredElements");
}
