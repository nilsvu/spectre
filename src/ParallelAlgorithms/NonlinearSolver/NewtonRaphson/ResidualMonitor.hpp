// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver::newton_raphson::detail {
template <typename Metavariables, typename FieldsTag>
struct InitializeResidualMonitor;
}  // namespace NonlinearSolver::newton_raphson::detail
/// \endcond

namespace NonlinearSolver::newton_raphson::detail {

template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags = tmpl::list<
      LinearSolver::Tags::Verbosity<OptionsGroup>,
      LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>,
      NonlinearSolver::Tags::SufficientDecreaseParameter<OptionsGroup>>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<InitializeResidualMonitor<FieldsTag, OptionsGroup>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              LinearSolver::observe_detail::Registration<OptionsGroup>>>>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void initialize(
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ResidualMonitor>(local_cache)
        .start_phase(next_phase);
  }
};

template <typename FieldsTag, typename OptionsGroup>
struct InitializeResidualMonitor {
 private:
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      LinearSolver::Tags::Initial<residual_magnitude_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags = db::AddSimpleTags<residual_magnitude_tag,
                                          initial_residual_magnitude_tag>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeResidualMonitor,
                                             simple_tags>(
            std::move(box),
            // The `UpdateResidual` action populates these tags with initial
            // values
            std::numeric_limits<double>::signaling_NaN(),
            std::numeric_limits<double>::signaling_NaN()),
        true);
  }
};

}  // namespace NonlinearSolver::newton_raphson::detail
