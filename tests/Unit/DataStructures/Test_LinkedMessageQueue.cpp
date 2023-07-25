// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/LinkedMessageQueue.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Label1;
struct Label2;

// Can't use "Queue" because Charm defines a type with that name.
template <typename Label>
struct MyQueue {
  // Non-copyable
  using type = std::unique_ptr<double>;
};

void test_queue() {
  LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>> queue{};
  CHECK(not queue.next_ready_id().has_value());

  queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
  CHECK(not queue.next_ready_id().has_value());
  queue.insert<MyQueue<Label2>>({1, {}}, std::make_unique<double>(-1.1));

  CHECK(queue.next_ready_id() == std::optional{1});
  {
    const auto out = queue.extract();
    CHECK(*tuples::get<MyQueue<Label1>>(out) == 1.1);
    CHECK(*tuples::get<MyQueue<Label2>>(out) == -1.1);
  }

  queue.insert<MyQueue<Label2>>({3, {1}}, std::make_unique<double>(-3.3));
  CHECK(not queue.next_ready_id().has_value());
  queue.insert<MyQueue<Label2>>({2, {3}}, std::make_unique<double>(-2.2));
  CHECK(not queue.next_ready_id().has_value());
  queue.insert<MyQueue<Label1>>({2, {3}}, std::make_unique<double>(2.2));

  const auto finish_checks = [](decltype(queue) test_queue) {
    CHECK(not test_queue.next_ready_id().has_value());
    test_queue.insert<MyQueue<Label1>>({3, {1}}, std::make_unique<double>(3.3));

    CHECK(test_queue.next_ready_id() == std::optional{3});
    {
      const auto out = test_queue.extract();
      CHECK(*tuples::get<MyQueue<Label1>>(out) == 3.3);
      CHECK(*tuples::get<MyQueue<Label2>>(out) == -3.3);
    }

    CHECK(test_queue.next_ready_id() == std::optional{2});
    {
      const auto out = test_queue.extract();
      CHECK(*tuples::get<MyQueue<Label1>>(out) == 2.2);
      CHECK(*tuples::get<MyQueue<Label2>>(out) == -2.2);
    }
  };
  finish_checks(serialize_and_deserialize(queue));
  finish_checks(std::move(queue));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.LinkedMessageQueue",
                  "[Unit][DataStructures]") {
  test_queue();

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>>
            queue{};
        queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
        queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
      }()),
      Catch::Matchers::Contains("Received duplicate messages at id 1 and "
                                "previous id --."));
  CHECK_THROWS_WITH(
      ([]() {
        LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>>
            queue{};
        queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
        queue.insert<MyQueue<Label2>>({2, {}}, std::make_unique<double>(1.1));
      }()),
      Catch::Matchers::Contains("Received messages with different ids (1 and "
                                "2) but the same previous id (--)."));
  CHECK_THROWS_WITH(
      ([]() {
        LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>>
            queue{};
        queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
        queue.extract();
      }()),
      Catch::Matchers::Contains(
          "Cannot extract before all messages have been received."));
#endif
}
