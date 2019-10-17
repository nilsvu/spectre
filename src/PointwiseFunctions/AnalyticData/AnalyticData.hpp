// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TypeTraits.hpp"

/// \cond
// Empty base class for marking analytic data.
struct MarkAsAnalyticData {};
/// \endcond

/// \ingroup AnalyticDataGroup
template <typename T>
using is_analytic_data = typename std::is_convertible<T*, MarkAsAnalyticData*>;

/// \ingroup AnalyticDataGroup
template <typename T>
constexpr bool is_analytic_data_v =
    cpp17::is_convertible_v<T*, MarkAsAnalyticData*>;
