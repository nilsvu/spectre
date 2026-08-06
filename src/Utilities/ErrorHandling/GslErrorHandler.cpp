// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/GslErrorHandler.hpp"

#include <gsl/gsl_errno.h>
#include <stdexcept>
#include <string>

void gsl_exception_handler(const char* reason, const char* file, const int line,
                           const int gsl_errno) {
  std::string msg = "GSL error: " + std::string(reason) +
                    " [errno=" + std::to_string(gsl_errno) + "]" + " at " +
                    file + ":" + std::to_string(line);
  throw std::runtime_error(msg);
}

void gsl_throw_exceptions() { gsl_set_error_handler(gsl_exception_handler); }
