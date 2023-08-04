// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

class OptionTest {
 public:
  OptionTest() = default;
  OptionTest(const OptionTest&) = default;
  OptionTest(OptionTest&&) = default;
  OptionTest& operator=(const OptionTest&) = default;
  OptionTest& operator=(OptionTest&&) = default;
  virtual ~OptionTest() = default;

  virtual std::string derived_name() const = 0;
};

struct OptionType {
  using type = std::unique_ptr<OptionTest>;
  static constexpr Options::String help = {"The type of OptionTest"};
};

class Test1 : public OptionTest {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {"A derived class"};
  Test1() = default;

  std::string derived_name() const override { return "Test1"; }
};

class Test2 : public OptionTest {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {""};
  Test2() = default;

  std::string derived_name() const override { return "Test2"; }
};

class TestWithArg : public OptionTest {
 public:
  struct Arg {
    using type = std::string;
    static constexpr Options::String help = {"halp"};
  };
  using options = tmpl::list<Arg>;
  static constexpr Options::String help = {""};
  TestWithArg() = default;
  explicit TestWithArg(std::string arg) : arg_(std::move(arg)) {}

  std::string derived_name() const override {
    return "TestWithArg(" + arg_ + ")";
  }

 private:
  std::string arg_;
};

// Same as TestWithArg, except there is an TestWithArg2::name() that
// returns something other than "TestWithArg2Arg" to test that class
// is named in the input file using pretty_type::name rather than
// pretty_type::short_name
class TestWithArg2 : public OptionTest {
 public:
  static std::string name() { return "ThisIsArg"; }
  struct Arg {
    using type = std::string;
    static constexpr Options::String help = {"halp"};
  };
  using options = tmpl::list<Arg>;
  static constexpr Options::String help = {""};
  TestWithArg2() = default;
  explicit TestWithArg2(std::string arg) : arg_(std::move(arg)) {}

  std::string derived_name() const override {
    return "TestWithArg2(" + arg_ + ")";
  }

 private:
  std::string arg_;
};

struct TestWithMetavars : OptionTest {
  struct Arg {
    using type = std::string;
    static constexpr Options::String help = {"halp"};
  };
  using options = tmpl::list<Arg>;
  static constexpr Options::String help = {""};

  TestWithMetavars() = default;
  template <typename Metavariables>
  explicit TestWithMetavars(std::string arg, const Options::Context& /*unused*/,
                            Metavariables /*meta*/)
      : arg_(std::move(arg)), valid_(Metavariables::valid) {}

  std::string derived_name() const override {
    return "TestWithArg(" + arg_ + ")" +
           (valid_ ? std::string{"yes"} : std::string{"no"});
  }

 private:
  std::string arg_;
  bool valid_{false};
};

class OtherBase {
 protected:
  OtherBase() = default;
  OtherBase(const OtherBase&) = default;
  OtherBase(OtherBase&&) = default;
  OtherBase& operator=(const OtherBase&) = default;
  OtherBase& operator=(OtherBase&&) = default;

 public:
  virtual ~OtherBase() = default;
};

struct OtherTag {
  using type = std::unique_ptr<OtherBase>;
  static constexpr Options::String help = {"An OtherBase"};
};

class OtherDerived : public OtherBase {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {""};
};

struct Vector {
  using type = std::vector<std::unique_ptr<OptionTest>>;
  static constexpr Options::String help = {"halp"};
};

struct Map {
  using type = std::map<std::string, std::unique_ptr<OptionTest>>;
  static constexpr Options::String help = {"halp"};
};

template <bool Valid>
struct Metavars {
  static constexpr bool valid = Valid;
  // [factory_creation]
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<OptionTest, tmpl::list<Test1, Test2, TestWithArg,
                                          TestWithArg2, TestWithMetavars>>,
        tmpl::pair<OtherBase, tmpl::list<OtherDerived>>>;
  };
  // [factory_creation]
  static_assert(tt::assert_conforms_to_v<factory_creation,
                                         Options::protocols::FactoryCreation>);
};

void test_factory() {
  {
    Options::Parser<tmpl::list<OptionType>> opts("");
    const std::string input = R"(
# [factory_without_arguments]
OptionType: Test2
# [factory_without_arguments]
)";
    opts.parse(input);
    CHECK(opts.get<OptionType, Metavars<true>>()->derived_name() == "Test2");
  }

  {
    Options::Parser<tmpl::list<OtherTag>> opts("");
    opts.parse("OtherTag: OtherDerived");
    // Just verify we got a result.  There's only one valid value that
    // can be returned anyway.
    CHECK(opts.get<OtherTag, Metavars<true>>());
  }
}

void test_factory_with_colon() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  Test2:");
  CHECK(opts.get<OptionType, Metavars<true>>()->derived_name() == "Test2");
}

void test_factory_with_arg() {
  Options::Parser<tmpl::list<OptionType>> opts("");
    const std::string input = R"(
# [factory_with_arguments]
OptionType:
  TestWithArg:
    Arg: stuff
# [factory_with_arguments]
)";
  opts.parse(input);
  CHECK(opts.get<OptionType, Metavars<true>>()->derived_name() ==
        "TestWithArg(stuff)");
}

void test_factory_with_name_function() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  ThisIsArg:\n"
      "    Arg: stuff");
  CHECK(opts.get<OptionType, Metavars<true>>()->derived_name() ==
        "TestWithArg2(stuff)");
}

void test_factory_with_metavars() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  TestWithMetavars:\n"
      "    Arg: stuff");
  CHECK(opts.get<OptionType, Metavars<true>>()->derived_name() ==
        "TestWithArg(stuff)yes");
  CHECK(opts.get<OptionType, Metavars<false>>()->derived_name() ==
        "TestWithArg(stuff)no");
  auto result_true = opts.apply<tmpl::list<OptionType>, Metavars<true>>(
      [&](auto arg) { return arg; });
  CHECK(result_true->derived_name() == "TestWithArg(stuff)yes");
  auto result_false = opts.apply<tmpl::list<OptionType>, Metavars<false>>(
      [&](auto arg) { return arg; });
  CHECK(result_false->derived_name() == "TestWithArg(stuff)no");
}

void test_factory_object_vector() {
  Options::Parser<tmpl::list<Vector>> opts("");
  opts.parse("Vector: [Test1, Test2, Test1]");
  const auto& arg = opts.get<Vector, Metavars<true>>();
  CHECK(arg.size() == 3);
  CHECK(arg[0]->derived_name() == "Test1");
  CHECK(arg[1]->derived_name() == "Test2");
  CHECK(arg[2]->derived_name() == "Test1");
}

void test_factory_object_map() {
  Options::Parser<tmpl::list<Map>> opts("");
  opts.parse(
      "Map:\n"
      "  A: Test1\n"
      "  B: Test2\n"
      "  C: Test1\n");
  const auto& arg = opts.get<Map, Metavars<true>>();
  CHECK(arg.size() == 3);
  CHECK(arg.at("A")->derived_name() == "Test1");
  CHECK(arg.at("B")->derived_name() == "Test2");
  CHECK(arg.at("C")->derived_name() == "Test1");
}

class IsCreatableBase {
 public:
  IsCreatableBase() = default;
  IsCreatableBase(const IsCreatableBase&) = default;
  IsCreatableBase(IsCreatableBase&&) = default;
  IsCreatableBase& operator=(const IsCreatableBase&) = default;
  IsCreatableBase& operator=(IsCreatableBase&&) = default;
  virtual ~IsCreatableBase() = default;
};

class NotCreatable : public IsCreatableBase {
 public:
  static constexpr bool factory_creatable = false;
  // Test with no `help` or `options`.
};

template <bool FactoryCreatable>
class MaybeCreatable : public IsCreatableBase {
 public:
  static constexpr bool factory_creatable = FactoryCreatable;
  static constexpr Options::String help = {"halp"};
  using options = tmpl::list<>;
};

// Hack to capture option parser errors
struct IsCreatableMessageHack {
  std::optional<std::string> message;
};
}  // namespace

template <>
struct Options::create_from_yaml<IsCreatableMessageHack> {
  template <typename Metavariables>
  static IsCreatableMessageHack create(const Options::Option& options) {
    try {
      options.parse_as<std::unique_ptr<IsCreatableBase>, Metavariables>();
      return {};
    } catch (const Options::detail::propagate_context& e) {
      return {e.message()};
    }
  }
};

namespace {
struct IsCreatableTag {
  using type = IsCreatableMessageHack;
  static constexpr Options::String help = {"halp"};
};

template <bool MaybeCreatableCreatable>
struct IsCreatableMetavars {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        IsCreatableBase,
        tmpl::list<NotCreatable, MaybeCreatable<MaybeCreatableCreatable>>>>;
  };
};

void test_factory_not_creatable() {
  const auto check_error_string = [](const auto creatable) {
    Options::Parser<tmpl::list<IsCreatableTag>> parser("");
    parser.parse("IsCreatableTag: BadValue");
    const auto error =
        parser.get<IsCreatableTag, IsCreatableMetavars<creatable>>();
    REQUIRE(error.message.has_value());
    const auto& message = *error.message;
    CAPTURE(message);
    CHECK(message.find("NotCreatable") == std::string::npos);
    CHECK((message.find("MaybeCreatable") != std::string::npos) == creatable);
  };
  check_error_string(std::true_type{});
  check_error_string(std::false_type{});

  const auto check_creation = [](const auto creatable) {
    Options::Parser<tmpl::list<IsCreatableTag>> parser("");
    parser.parse("IsCreatableTag: MaybeCreatable");
    const auto error =
        parser.get<IsCreatableTag, IsCreatableMetavars<creatable>>();
    REQUIRE(error.message.has_value() != creatable);
    if (not creatable) {
      CAPTURE(*error.message);
      CHECK(error.message->find("Unknown Id 'MaybeCreatable'") !=
            std::string::npos);
    }
  };
  check_creation(std::true_type{});
  check_creation(std::false_type{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Factory", "[Unit][Options]") {
  test_factory();
  test_factory_with_arg();
  test_factory_with_name_function();
  test_factory_with_colon();
  test_factory_with_metavars();
  test_factory_object_vector();
  test_factory_object_map();
  test_factory_not_creatable();
}

namespace {
void trigger_missing() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:");
  opts.get<OptionType, Metavars<true>>();
}

void trigger_multiple() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  Test1:\n"
      "  Test2:");
  opts.get<OptionType, Metavars<true>>();
}

void trigger_vector() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: []");
  opts.get<OptionType, Metavars<true>>();
}

void trigger_unknown() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: Potato");
  opts.get<OptionType, Metavars<true>>();
}

void trigger_missing_arg() {
  Options::Parser<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  TestWithArg:");
  opts.get<OptionType, Metavars<true>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Factory.missing", "[Unit][Options]") {
  CHECK_THROWS_WITH(trigger_missing(),
                    Catch::Matchers::ContainsSubstring(
                        "At line 1 column 1:\nExpected a class to "
                        "create:\nKnown Ids:\n  Test1"));
}

SPECTRE_TEST_CASE("Unit.Options.Factory.multiple", "[Unit][Options]") {
  CHECK_THROWS_WITH(
      trigger_multiple(),
      Catch::Matchers::ContainsSubstring(
          "At line 2 column 3:\nExpected a single class to create, got 2"));
}

SPECTRE_TEST_CASE("Unit.Options.Factory.vector", "[Unit][Options]") {
  CHECK_THROWS_WITH(
      trigger_vector(),
      Catch::Matchers::ContainsSubstring(
          "At line 1 column 13:\nExpected a class or a class with options"));
}

SPECTRE_TEST_CASE("Unit.Options.Factory.unknown", "[Unit][Options]") {
  CHECK_THROWS_WITH(trigger_unknown(),
                    Catch::Matchers::ContainsSubstring(
                        "At line 1 column 13:\nUnknown Id 'Potato'"));
}

SPECTRE_TEST_CASE("Unit.Options.Factory.missing_arg", "[Unit][Options]") {
  CHECK_THROWS_WITH(
      trigger_missing_arg(),
      Catch::Matchers::ContainsSubstring(
          "At line 2 column 1:\nYou did not specify the option (Arg)"));
}
