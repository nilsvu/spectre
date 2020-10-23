// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace NonlinearSolver::newton_raphson {
/*!
 * \brief Find the next step length for the line-search globalization
 *
 * The step length is chosen such that it minimizes the quadratic (first
 * globalization step) or cubic (subsequent globalization steps) polynomial
 * interpolation. This function implements Algorithm A6.1.3 in
 * \cite DennisSchnabel (p. 325).
 */
double line_search(size_t globalization_iteration_id, double step_length,
                   double prev_step_length, double residual,
                   double residual_slope, double next_residual,
                   double prev_residual);
}  // namespace NonlinearSolver::newton_raphson
