// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Element.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {

template <typename InboxTag, size_t Dim, typename TemporalIdType,
          typename... InboxTags>
bool has_received_from_all_mortars(
    const TemporalIdType& temporal_id, const Element<Dim>& element,
    const tuples::TaggedTuple<InboxTags...>& inboxes) noexcept {
  if (element.number_of_neighbors() == 0) {
    return true;
  }
  const auto& inbox = tuples::get<InboxTag>(inboxes);
  const auto temporal_received = inbox.find(temporal_id);
  if (temporal_received == inbox.end()) {
    return false;
  }
  const auto& received_neighbor_data = temporal_received->second;
  for (const auto& direction_and_neighbors : element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    for (const auto& neighbor : direction_and_neighbors.second) {
      const auto neighbor_received =
          received_neighbor_data.find(MortarId<Dim>{direction, neighbor});
      if (neighbor_received == received_neighbor_data.end()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace dg
