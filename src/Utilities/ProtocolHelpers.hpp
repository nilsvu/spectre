// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

// Note that std::is_convertible is used in the following type aliases as it
// will not match private base classes (unlike std::is_base_of)

// @{
/*
 * \brief Checks if the `ConformingType` conforms to the `Protocol`.
 *
 * By default, only checks if the class derives off the protocol to reduce
 * compile time. Protocol conformance is tested rigorously in the unit tests
 * instead. Set the `SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE` CMake option to
 * always enable rigorous protocol conformance checks.
 *
 * \see \ref protocols
 */
#ifdef SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
template <typename ConformingType, typename Protocol>
constexpr bool conforms_to_v =
    cpp17::is_convertible_v<ConformingType*, Protocol*> and
        Protocol::template is_conforming_v<ConformingType>;
template <typename ConformingType, typename Protocol>
using conforms_to =
    cpp17::bool_constant<conforms_to_v<ConformingType, Protocol>>;
#else   // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
template <typename ConformingType, typename Protocol>
using conforms_to = typename std::is_convertible<ConformingType*, Protocol*>;
template <typename ConformingType, typename Protocol>
constexpr bool conforms_to_v =
    cpp17::is_convertible_v<ConformingType*, Protocol*>;
#endif  // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
// @}
