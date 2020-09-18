// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"

namespace logging::Actions {

template <typename Formatter>
struct Log {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename... ReductionTypes>
  static void apply(const db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    ReductionTypes&&... reduction_data) noexcept {
    Parallel::printf(Formatter::apply(std::move(reduction_data)...) + "\n");
  }
};

}  // namespace logging::Actions
