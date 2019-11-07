// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/ProtocolTestHelpers.hpp"

namespace {

/// [named_protocol]
namespace protocols {
/*!
 * \brief Has a name.
 *
 * Requires the class has these member functions:
 * - `name`: Returns the name of the object as a `std::string`.
 */
struct Named {
  CREATE_IS_CALLABLE(name)
  template <typename ConformingType>
  static constexpr bool is_conforming_v =
      is_name_callable_r_v<std::string, ConformingType>;
};
}  // namespace protocols
/// [named_protocol]

/// [named_conformance]
class Person : public protocols::Named {
 public:
  // Function required to conform to the protocol
  std::string name() const { return first_name_ + " " + last_name_; }

 private:
  // Implementation details of the class that are irrelevant to the protocol
  std::string first_name_;
  std::string last_name_;

 public:
  Person(std::string first_name, std::string last_name)
      : first_name_(std::move(first_name)), last_name_(std::move(last_name)) {}
};
/// [named_conformance]

/// [using_named_protocol]
template <typename NamedThing>
std::string greet(const NamedThing& named_thing) {
  // Make sure the template parameter conforms to the protocol
  static_assert(conforms_to_v<NamedThing, protocols::Named>,
                "NamedThing must be Named.");
  // Now we can rely on the interface that the protocol defines
  return "Hello, " + named_thing.name() + "!";
}
/// [using_named_protocol]

}  // namespace

// Test conforms_to metafunction
/// [conforms_to]
static_assert(conforms_to_v<Person, protocols::Named>,
              "Failed testing conforms_to_v");
/// [conforms_to]
namespace {
struct NotNamed {};
class DerivedNonConformingClass : private Person {};
class DerivedConformingClass : public Person {};
}  // namespace
static_assert(not conforms_to_v<NotNamed, protocols::Named>,
              "Failed testing conforms_to_v");
static_assert(not conforms_to_v<DerivedNonConformingClass, protocols::Named>,
              "Failed testing conforms_to_v");
static_assert(conforms_to_v<DerivedConformingClass, protocols::Named>,
              "Failed testing conforms_to_v");

// Give an example how a protocol author should test a new protocol
/// [testing_a_protocol]
static_assert(protocols::Named::template is_conforming_v<Person>,
              "Failed testing the protocol");
static_assert(not protocols::Named::template is_conforming_v<NotNamed>,
              "Failed testing the protocol");
/// [testing_a_protocol]

// Give an example how protocol consumers should test protocol conformance
/// [test_protocol_conformance]
static_assert(test_protocol_conformance<Person, protocols::Named>,
              "Failed testing protocol conformance");
/// [test_protocol_conformance]
