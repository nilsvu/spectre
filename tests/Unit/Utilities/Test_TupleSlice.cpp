// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "Utilities/TupleSlice.hpp"

namespace {

struct MoveTest {
  MoveTest() = default;
  MoveTest(const MoveTest& /*rhs*/) noexcept { status = "copy-constructed"; }
  MoveTest& operator=(const MoveTest& /*rhs*/) noexcept {
    status = "copied";
    return *this;
  };
  MoveTest(MoveTest&& rhs) noexcept {
    status = "move-constructed";
    rhs.status = "move-constructed-away";
  }
  MoveTest& operator=(MoveTest&& rhs) noexcept {
    status = "moved";
    rhs.status = "moved-away";
    return *this;
  };
  ~MoveTest() = default;

  std::string status = "initial";
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.TupleSlice", "[Utilities][Unit]") {
  {
    std::tuple<int, float, double, std::string> tuple{1, 2., 3., ""};
    CHECK(tuple_slice<1, 3>(tuple) == std::tuple<float, double>{2., 3.});
    CHECK(tuple_slice<1, 3>(std::move(tuple)) ==
          std::tuple<float, double>{2., 3.});
  }
  {
    std::tuple<int, float, double, std::string> tuple{1, 2., 3., ""};
    CHECK(tuple_head<2>(tuple) == std::tuple<int, float>{1, 2.});
    CHECK(tuple_head<2>(std::move(tuple)) == std::tuple<int, float>{1, 2.});
  }
  {
    std::tuple<int, float, double, std::string> tuple{1, 2., 3., ""};
    CHECK(tuple_tail<2>(tuple) == std::tuple<double, std::string>{3., ""});
    CHECK(tuple_tail<2>(std::move(tuple)) ==
          std::tuple<double, std::string>{3., ""});
  }
  CHECK(tuple_slice<0, 1>(std::tuple<int>{1}) == std::tuple<int>{1});
  CHECK(tuple_head<1>(std::tuple<int>{1}) == std::tuple<int>{1});
  CHECK(tuple_tail<1>(std::tuple<int>{1}) == std::tuple<int>{1});
  CHECK(tuple_slice<0, 0>(std::tuple<int>{1}) == std::tuple<>{});
  CHECK(tuple_head<0>(std::tuple<int>{1}) == std::tuple<>{});
  CHECK(tuple_tail<0>(std::tuple<int>{1}) == std::tuple<>{});
  CHECK(tuple_slice<0, 0>(std::tuple<>{}) == std::tuple<>{});
  CHECK(tuple_head<0>(std::tuple<>{}) == std::tuple<>{});
  CHECK(tuple_tail<0>(std::tuple<>{}) == std::tuple<>{});
  {
    INFO("slicing should reference");
    MoveTest a{};
    MoveTest& b = a;
    const std::tuple<MoveTest, MoveTest&, MoveTest&&> tuple{a, b, MoveTest{}};
    CHECK(std::get<0>(tuple).status == "copy-constructed");
    CHECK(std::get<1>(tuple).status == "initial");
    CHECK(std::get<2>(tuple).status == "initial");
    const auto sliced_tuple = tuple_slice<0, 3>(tuple);
    CHECK(b.status == "initial");
    CHECK(std::get<0>(sliced_tuple).status == "copy-constructed");
    CHECK(&std::get<0>(sliced_tuple) == &std::get<0>(tuple));
    CHECK(std::get<1>(sliced_tuple).status == "initial");
    CHECK(&std::get<1>(sliced_tuple) == &std::get<1>(tuple));
    CHECK(std::get<2>(sliced_tuple).status == "initial");
    CHECK(&std::get<2>(sliced_tuple) == &std::get<2>(tuple));
  }
  {
    INFO("moving the tuple");
    MoveTest a{};
    MoveTest& b = a;
    std::tuple<MoveTest, MoveTest&, MoveTest&&> tuple{a, b, MoveTest{}};
    CHECK(std::get<0>(tuple).status == "copy-constructed");
    std::get<0>(tuple).status = "initial";
    CHECK(std::get<1>(tuple).status == "initial");
    CHECK(std::get<2>(tuple).status == "initial");
    const auto sliced_tuple = tuple_slice<0, 3>(std::move(tuple));
    CHECK(b.status == "initial");
    CHECK(std::get<0>(sliced_tuple).status == "initial");
    CHECK(std::get<1>(sliced_tuple).status == "initial");
    CHECK(std::get<2>(sliced_tuple).status == "initial");
  }
  {
    INFO("works with non-tuple containers");
    CHECK(tuple_head<1>(std::array<double, 2>{{1., 2.}}) ==
          std::tuple<double>{{1.}});
    CHECK(tuple_head<1>(std::pair<int, float>{1, 2.}) == std::tuple<int>{1});
  }
}
