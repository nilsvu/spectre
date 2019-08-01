// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver {
namespace newton_raphson_detail {
template <typename Metavariables, typename FieldsTag>
struct InitializeResidualMonitor;
}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

template <typename Metavariables, typename FieldsTag>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list =
      tmpl::list<NonlinearSolver::OptionTags::Verbosity,
                 NonlinearSolver::OptionTags::ConvergenceCriteria>;
  using options = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<InitializeResidualMonitor<Metavariables, FieldsTag>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              NonlinearSolver::observe_detail::Registration>>>>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ResidualMonitor>(local_cache)
        .start_phase(next_phase);
  }
};

template <typename Metavariables, typename FieldsTag>
struct InitializeResidualMonitor {
 private:
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag = db::add_tag_prefix<
      ::Tags::Initial,
      db::add_tag_prefix<
          LinearSolver::Tags::Magnitude,
          db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags = db::AddSimpleTags<NonlinearSolver::Tags::IterationId,
                                          residual_magnitude_tag,
                                          initial_residual_magnitude_tag>;
    using compute_tags = db::AddComputeTags<
        NonlinearSolver::Tags::HasConvergedCompute<fields_tag>>;

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeResidualMonitor,
                                             simple_tags, compute_tags>(
            std::move(box),
            db::item_type<NonlinearSolver::Tags::IterationId>{0},
            std::numeric_limits<double>::signaling_NaN(),
            std::numeric_limits<double>::signaling_NaN()),
        true);
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
