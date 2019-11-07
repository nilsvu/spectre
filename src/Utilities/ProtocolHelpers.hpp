// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

// Note that std::is_convertible is used in the following type aliases as it
// will not match private base classes (unlike std::is_base_of)

// @{
/// Checks if the `ConformingType` conforms to the `Protocol`.
///
/// See \ref protocols
template <typename ConformingType, typename Protocol>
using conforms_to = typename std::is_convertible<ConformingType*, Protocol*>;

template <typename ConformingType, typename Protocol>
constexpr bool conforms_to_v =
    cpp17::is_convertible_v<ConformingType*, Protocol*>;
// @}
