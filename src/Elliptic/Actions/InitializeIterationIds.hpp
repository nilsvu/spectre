// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/IterationId.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"

namespace elliptic {
namespace Actions {

/*!
 * \brief Initializes DataBox tags for the elliptic iteration IDs.
 *
 * This action simply constructs all `ComponentTags` from zero integers.
 * This is suitable for elliptic iteration IDs that represent step numbers.
 * It also adds a compute tag for the `::Tags::Next` of each of the
 * `ComponentTags`, as well as for the `elliptic::Tags::IterationId` that
 * combines all components into a single number.
 *
 * DataBox:
 * - Adds:
 *   - `ComponentTags...`
 *   - `Tags::Next<ComponentTags>...`
 *   - `elliptic::Tags::IterationId`
 */
template <typename... ComponentTags>
struct InitializeIterationIds {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags = db::AddSimpleTags<ComponentTags...>;
    using compute_tags = db::AddComputeTags<
        ::Tags::NextCompute<ComponentTags>...,
        elliptic::Tags::IterationIdCompute<ComponentTags...>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeIterationIds,
                                             simple_tags, compute_tags>(
            std::move(box), db::item_type<ComponentTags>{0}...));
  }
};

}  // namespace Actions
}  // namespace elliptic
