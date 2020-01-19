// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace evolution {

/// Empty base class for marking a class as numeric initial data.
///
/// \see `evolution::is_numeric_initial_data`
struct MarkAsNumericInitialData {};

// @{
/// Checks if the class `T` is marked as numeric initial data.
template <typename T>
using is_numeric_initial_data =
    typename std::is_convertible<T*, MarkAsNumericInitialData*>;

template <typename T>
constexpr bool is_numeric_initial_data_v =
    cpp17::is_convertible_v<T*, MarkAsNumericInitialData*>;
// @}

/// Provides compile-time information for importing numeric initial data for
/// the `System` from a volume data file.
template <typename System>
struct NumericInitialData : MarkAsNumericInitialData {
  using import_fields =
      db::get_variables_tags_list<typename System::variables_tag>;
};

}  // namespace evolution
