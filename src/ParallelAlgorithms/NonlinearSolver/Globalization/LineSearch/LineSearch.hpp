// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/NonlinearSolver/Globalization/LineSearch/ElementActions.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Globalization/LineSearch/InitializeElement.hpp"

namespace NonlinearSolver {
namespace Globalization {

struct LineSearch {
  using initialize_element = LineSearch_detail::InitializeElement;
  using prepare = LineSearch_detail::Prepare;
  using perform_step = LineSearch_detail::PerformStep;
};

}  // namespace Globalization
}  // namespace NonlinearSolver
