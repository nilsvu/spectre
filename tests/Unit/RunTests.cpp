// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "RunTests.hpp"

#include "Framework/TestingFramework.hpp"

#include <charm++.h>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <string>

#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/System/Abort.hpp"
#include "Utilities/System/Exit.hpp"
#include "tests/Unit/RunTestsRegister.hpp"

RunTests::RunTests(CkArgMsg* msg) {
  setup_error_handling();
  setup_memory_allocation_failure_reporting();
  register_run_tests_libs();
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  enable_segfault_handler();
  Catch::StringMaker<double>::precision =
      std::numeric_limits<double>::max_digits10;
  Catch::StringMaker<float>::precision =
      std::numeric_limits<float>::max_digits10;
  const int result = Catch::Session().run(msg->argc, msg->argv);
  // In the case where we run all the non-failure tests at once we must ensure
  // that we only initialize and finalize the python env once. Initialization is
  // done in the constructor of SetupLocalPythonEnvironment, while finalization
  // is done in the constructor of RunTests.
  pypp::SetupLocalPythonEnvironment::finalize_env();
  if (0 == result) {
    sys::exit();
  }
  sys::abort("A catch test has failed.");
}

#include "tests/Unit/RunTests.def.h"

// Needed for tests that use the GlobalCache since it registers itself with
// Charm++. However, since Parallel/CharmMain.tpp isn't included in the RunTests
// executable, no actual registration is done, the GlobalCache is only
// queued for registration.
namespace Parallel::charmxx {
class RegistrationHelper;
std::unique_ptr<RegistrationHelper>* charm_register_list = nullptr;
size_t charm_register_list_capacity = 0;
size_t charm_register_list_size = 0;
}  // namespace Parallel::charmxx
