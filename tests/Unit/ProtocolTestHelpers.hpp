// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/ProtocolHelpers.hpp"

namespace detail {

template <typename ConformingType, typename Protocol>
struct TestProtocolConformanceImpl : std::true_type {
  static_assert(
      conforms_to_v<ConformingType, Protocol>,
      "The type does not indicate it conforms to the protocol. The protocol is "
      "listed as the first template parameter to `test_protocol_conformance` "
      "and the conforming type is listed as the second template parameter. "
      "Have you forgotten to (publicly) derive it off the protocol?");
  static_assert(
      Protocol::template is_conforming_v<ConformingType>,
      "The type does not conforms to the protocol. The protocol is "
      "listed as the first template parameter to `test_protocol_conformance` "
      "and the conforming type is listed as the second template parameter.");
};

}  // namespace detail

template <typename ConformingType, typename Protocol>
constexpr bool test_protocol_conformance =
    detail::TestProtocolConformanceImpl<ConformingType, Protocol>::value;
