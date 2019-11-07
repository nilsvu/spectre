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
              "The class does not conform to the protocol.");
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

// Give examples about protocol antipatterns
namespace {
namespace protocols {
/// [named_antipattern]
// Don't do this. Protocols should not be templates.
template <typename NameType>
struct NamedAntipattern {
  CREATE_IS_CALLABLE(name)
  // Also checking that the `name` function returns a particular type
  template <typename ConformingType>
  static constexpr bool is_conforming_v =
      is_name_callable_r_v<NameType, ConformingType>;
};
/// [named_antipattern]
// Just making sure the protocol works correctly, even though this is an
// antipattern
static_assert(NamedAntipattern<std::string>::template is_conforming_v<Person>,
              "Failed testing is_conforming_v");
static_assert(not NamedAntipattern<int>::template is_conforming_v<Person>,
              "Failed testing is_conforming_v");
/// [named_with_type]
// Instead, do this.
struct NamedWithType {
  CREATE_HAS_TYPE_ALIAS(NameType)
  CREATE_IS_CALLABLE(name)
  // Lazily evaluated so we can use `ConformingType::NameType`
  template <typename ConformingType>
  struct IsNameCallable
      : is_name_callable_r<typename ConformingType::NameType, ConformingType> {
  };
  // First check the class has a `NameType`, then use it to check the return
  // type of the `name` function.
  template <typename ConformingType>
  static constexpr bool is_conforming_v =
      std::conditional_t<has_NameType_v<ConformingType>,
                         IsNameCallable<ConformingType>,
                         std::false_type>::value;
};
/// [named_with_type]
}  // namespace protocols
/// [person_with_name_type]
struct PersonWithNameType : protocols::NamedWithType {
  using NameType = std::string;
  std::string name() const;
};
/// [person_with_name_type]
// Make sure the protocol is implemented correctly
static_assert(not protocols::NamedWithType::template is_conforming_v<Person>,
              "Failed testing is_conforming_v");
static_assert(
    protocols::NamedWithType::template is_conforming_v<PersonWithNameType>,
    "Failed testing is_conforming_v");
/// [example_check_name_type]
static_assert(conforms_to_v<PersonWithNameType, protocols::NamedWithType>,
              "The class does not conform to the protocol.");
static_assert(
    cpp17::is_same_v<typename PersonWithNameType::NameType, std::string>,
    "The `NameType` isn't a `std::string`!");
/// [example_check_name_type]
}  // namespace

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
