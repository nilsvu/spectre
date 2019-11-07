// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/ProtocolHelpers.hpp"

/*!
 * \ingroup ProtocolsGroup
 * \brief Test that the `ConformingType` conforms to the `Protocol`
 *
 * Since the `conforms_to_v` metafunction only checks if a class _indicates_
 * it conforms to the protocol (unless the
 * `SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE` flag is set), use the
 * `test_protocol_conformance` metafunction in unit tests to check the class
 * actually fulfills the requirements defined by the protocol's
 * `is_conforming_v` metafunction.
 */
template <typename ConformingType, typename Protocol>
struct test_protocol_conformance {
#ifdef SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
  static_assert(
      conforms_to_v<ConformingType, Protocol>,
      "The type does not conform to the protocol or does not (publicly) derive "
      "off it. The type is listed as the first template parameter to "
      "`test_protocol_conformance` and the protocol is listed as the second "
      "template parameter.");
#elif   // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
  static_assert(
      conforms_to_v<ConformingType, Protocol>,
      "The type does not indicate it conforms to the protocol. The type is "
      "listed as the first template parameter to `test_protocol_conformance` "
      "and the protocol is listed as the second template parameter. "
      "Have you forgotten to (publicly) derive it off the protocol?");
  static_assert(
      Protocol::template is_conforming_v<ConformingType>,
      "The type does not conform to the protocol. The type is "
      "listed as the first template parameter to `test_protocol_conformance` "
      "and the protocol is listed as the second template parameter.");
#endif  // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
};
