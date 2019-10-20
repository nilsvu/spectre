// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
// Empty base class for marking analytic solutions.
struct MarkAsAnalyticSolution {};
/// \endcond

/// \ingroup AnalyticSolutionsGroup
template <typename T>
using is_analytic_solution =
    typename std::is_convertible<T*, MarkAsAnalyticSolution*>;

/// \ingroup AnalyticSolutionsGroup
template <typename T>
constexpr bool is_analytic_solution_v =
    cpp17::is_convertible_v<T*, MarkAsAnalyticSolution*>;
