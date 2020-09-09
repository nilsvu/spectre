// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... InboxTags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Parallel {
namespace Actions {

template <typename InboxTag, bool EnableIfDisabled = false,
          size_t TemporalIdIndex = 0>
struct ReceiveData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename... DataTypes>
  static void apply(const db::DataBox<DbTagsList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    DataTypes... data) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    auto data_tuple = std::make_tuple(std::move(data)...);
    Parallel::receive_data<InboxTag>(receiver_proxy,
                                     get<TemporalIdIndex>(data_tuple),
                                     std::move(data_tuple), EnableIfDisabled);
  }
};

}  // namespace Actions
}  // namespace Parallel
