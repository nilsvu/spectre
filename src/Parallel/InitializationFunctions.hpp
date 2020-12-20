// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <exception>

#include "ErrorHandling/Abort.hpp"

inline void setup_error_handling() {
  std::set_terminate([]() {
    abort(
        "Terminate was called, calling Charm++'s abort function to properly "
        "terminate execution.");
  });
}
