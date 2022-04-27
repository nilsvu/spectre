// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace NonlinearSolver {

struct Damping {
  struct Factor {
    using type = double;
    static constexpr Options::String help =
        "Multiply corrections by this factor";
  };
  struct Iterations {
    using type = size_t;
    static constexpr Options::String help = "Damp this number of iterations";
  };
  static constexpr Options::String help = "Damping options";
  using options = tmpl::list<Factor, Iterations>;
  double factor;
  size_t num_iterations;
  void pup(PUP::er& p) {
    p | factor;
    p | num_iterations;
  }
};

}  // namespace NonlinearSolver
