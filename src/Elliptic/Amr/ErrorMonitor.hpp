// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Elliptic/Amr/Observe.hpp"
#include "Elliptic/Amr/Tags.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace elliptic::amr::detail {
template <size_t Dim>
struct InitializeErrorMonitor;
}  // namespace elliptic::amr::detail
/// \endcond

namespace elliptic::amr::detail {

template <typename Metavariables, size_t Dim>
struct ErrorMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionTags::AmrGroup>,
                 Convergence::Tags::Criteria<OptionTags::AmrGroup>>;
  // The actions in `ErrorMonitorActions.hpp` are invoked as simple actions on
  // this component as the result of reductions from the actions in
  // `ElementActions.hpp`. See `elliptic::amr::Amr` for details.
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<::Actions::SetupDataBox, InitializeErrorMonitor<Dim>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              observe_detail::Registration>>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ErrorMonitor>(local_cache)
        .start_phase(next_phase);
  }
};

template <size_t Dim>
struct InitializeErrorMonitor {
  using simple_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), Parallel::AlgorithmExecution::Halt};
  }
};

}  // namespace elliptic::amr::detail
