// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <csignal>

#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

SPECTRE_TEST_CASE("Unit.ErrorHandling.SegfaultHandler",
                  "[ErrorHandling][Unit]") {
  enable_segfault_handler();
  CHECK_THROWS_WITH(std::raise(SIGSEGV),
                    Catch::Matchers::ContainsSubstring("Segmentation fault!"));
}
