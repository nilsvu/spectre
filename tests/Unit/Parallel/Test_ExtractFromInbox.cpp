// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/ExtractFromInbox.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct TemporalIdTag : db::SimpleTag {
  using type = size_t;
};

struct SampleDataTag {
  using type = std::map<size_t, int>;
};

SPECTRE_TEST_CASE("Unit.Parallel.ExtractFromInbox", "[Parallel][Unit]") {
  const size_t temporal_id = 0;
  const auto box = db::create<db::AddSimpleTags<TemporalIdTag>>(temporal_id);
  tuples::TaggedTuple<SampleDataTag> inboxes{};
  tuples::get<SampleDataTag>(inboxes).emplace(temporal_id, 1);
  CHECK(Parallel::extract_from_inbox<SampleDataTag, TemporalIdTag>(inboxes,
                                                                   box) == 1);
  CHECK(tuples::get<SampleDataTag>(inboxes).size() == 0);
}
}  // namespace
