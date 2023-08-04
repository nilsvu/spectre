// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_set>

#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
void test_errors() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(SegmentId(3, 8), Catch::Matchers::ContainsSubstring(
                                         "index = 8, refinement_level = 3"));
  CHECK_THROWS_WITH(
      SegmentId(0, 0).id_of_parent(),
      Catch::Matchers::ContainsSubstring("on root refinement level!"));
  CHECK_THROWS_WITH(
      SegmentId(0, 0).id_of_sibling(),
      Catch::Matchers::ContainsSubstring(
          "The segment on the root refinement level has no sibling"));
  CHECK_THROWS_WITH(
      SegmentId(0, 0).id_of_abutting_nibling(),
      Catch::Matchers::ContainsSubstring(
          "The segment on the root refinement level has no abutting nibling"));
  CHECK_THROWS_WITH(
      SegmentId(0, 0).side_of_sibling(),
      Catch::Matchers::ContainsSubstring(
          "The segment on the root refinement level has no sibling"));
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.SegmentId", "[Domain][Unit]") {
  // Test equality operator:
  SegmentId segment_one(4, 3);
  SegmentId segment_two(4, 3);
  SegmentId segment_three(4, 0);
  SegmentId segment_four(5, 4);
  CHECK(segment_one == segment_two);
  CHECK(segment_two != segment_three);
  CHECK(segment_two != segment_four);

  // Test pup operations:
  test_serialization(segment_one);

  // Test parent and child operations:
  for (size_t level = 0; level < 5; ++level) {
    const double segment_length = 2.0 / two_to_the(level);
    double midpoint = -1.0 + 0.5 * segment_length;
    for (size_t segment_index = 0; segment_index < two_to_the(level);
         ++segment_index) {
      SegmentId id(level, segment_index);
      CHECK(id.midpoint() == midpoint);
      CHECK((id.endpoint(Side::Upper) + id.endpoint(Side::Lower)) / 2. ==
            midpoint);
      CHECK(id.endpoint(Side::Upper) - id.endpoint(Side::Lower) ==
            segment_length);
      midpoint += segment_length;
      CHECK(id == id.id_of_child(Side::Lower).id_of_parent());
      CHECK(id == id.id_of_child(Side::Upper).id_of_parent());
      CHECK(id.overlaps(id));
      if (0 != level) {
        CHECK(id.overlaps(id.id_of_parent()));
        const Side side_of_parent =
            0 == segment_index % 2 ? Side::Lower : Side::Upper;
        CHECK(id == id.id_of_parent().id_of_child(side_of_parent));
        CHECK_FALSE(id.overlaps(
            id.id_of_parent().id_of_child(opposite(side_of_parent))));
      }
      CHECK(id.id_of_child(Side::Lower).id_of_sibling() ==
            id.id_of_child(Side::Upper));
      CHECK(id.id_of_child(Side::Upper).id_of_sibling() ==
            id.id_of_child(Side::Lower));
      CHECK(id.id_of_child(Side::Lower).id_of_abutting_nibling() ==
            id.id_of_child(Side::Upper).id_of_child(Side::Lower));
      CHECK(id.id_of_child(Side::Upper).id_of_abutting_nibling() ==
            id.id_of_child(Side::Lower).id_of_child(Side::Upper));
      CHECK(id.id_of_child(Side::Lower).side_of_sibling() == Side::Upper);
      CHECK(id.id_of_child(Side::Upper).side_of_sibling() == Side::Lower);
    }
  }

  // Test retrieval functions:
  SegmentId level_2_index_3(2, 3);
  CHECK(level_2_index_3.refinement_level() == 2);
  CHECK(level_2_index_3.index() == 3);

  // Test output operator:
  SegmentId level_3_index_2(3, 2);
  CHECK(get_output(level_3_index_2) == "L3I2");

  {
    INFO("Hash");
    CHECK(std::unordered_set<SegmentId>{SegmentId{2, 3}, SegmentId{2, 3},
                                        SegmentId{3, 2}} ==
          std::unordered_set<SegmentId>{SegmentId{3, 2}, SegmentId{2, 3}});
  }

  test_errors();
}
