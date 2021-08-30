// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Elliptic/Amr/InboxTags.hpp"
#include "Elliptic/Amr/Observe.hpp"
#include "Elliptic/Amr/Tags.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace elliptic::amr::detail {

template <size_t Dim, typename BroadcastTarget>
struct MeasureError {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<
                Convergence::Tags::Criteria<OptionTags::AmrGroup>, DataBox>> =
                nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const size_t level) noexcept {
    const double error = 1.;
    const double initial_error = 1.;
    observe_detail::contribute_to_reduction_observer<ParallelComponent>(
        level, error, cache);

    // Determine whether AMR has converged
    Convergence::HasConverged has_converged{
        get<Convergence::Tags::Criteria<OptionTags::AmrGroup>>(box), level,
        error, initial_error};

    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionTags::AmrGroup>>(box) >=
                 ::Verbosity::Quiet)) {
      Parallel::printf("AMR(%zu) complete. Remaining error: %e\n", level,
                       error);
    }
    if (UNLIKELY(has_converged and
                 get<logging::Tags::Verbosity<OptionTags::AmrGroup>>(cache) >=
                     ::Verbosity::Quiet)) {
      Parallel::printf("AMR has converged in %zu iterations: %s\n", level,
                       has_converged);
    }

    Parallel::receive_data<InboxTags::ErrorMeasurement>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), level,
        std::make_tuple(error, std::move(has_converged)));
  }
};

}  // namespace elliptic::amr::detail
