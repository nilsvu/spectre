// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver {
namespace newton_raphson_detail {
template <typename Metavariables>
struct InitializeResidualMonitor;
}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

template <typename Metavariables>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list =
      tmpl::list<NonlinearSolver::OptionTags::Verbosity,
                 NonlinearSolver::OptionTags::ConvergenceCriteria>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<tmpl::append<
      typename InitializeResidualMonitor<Metavariables>::simple_tags,
      typename InitializeResidualMonitor<Metavariables>::compute_tags>>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    Parallel::simple_action<InitializeResidualMonitor<Metavariables>>(
        Parallel::get_parallel_component<ResidualMonitor>(
            *(global_cache.ckLocalBranch())));

    const auto initial_observation_id = observers::ObservationId(
        db::item_type<NonlinearSolver::Tags::IterationId>{0},
        typename NonlinearSolver::observe_detail::ObservationType{});
    Parallel::simple_action<
        observers::Actions::RegisterSingletonWithObserverWriter>(
        Parallel::get_parallel_component<ResidualMonitor>(
            *(global_cache.ckLocalBranch())),
        initial_observation_id);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};

template <typename Metavariables>
struct InitializeResidualMonitor {
 private:
  using fields_tag = typename Metavariables::system::nonlinear_fields_tag;
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag = db::add_tag_prefix<
      ::Tags::Initial,
      db::add_tag_prefix<
          LinearSolver::Tags::Magnitude,
          db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>>;

 public:
  using simple_tags =
      db::AddSimpleTags<NonlinearSolver::Tags::IterationId,
                        residual_magnitude_tag, initial_residual_magnitude_tag,
                        // Need to place this in the DataBox to make it
                        // available to compute items
                        NonlinearSolver::OptionTags::ConvergenceCriteria>;
  using compute_tags = db::AddComputeTags<
      NonlinearSolver::Tags::HasConvergedCompute<fields_tag>>;

  template <typename... InboxTags, typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto box = db::create<simple_tags, compute_tags>(
        db::item_type<NonlinearSolver::Tags::IterationId>{0},
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        get<NonlinearSolver::OptionTags::ConvergenceCriteria>(cache));
    return std::make_tuple(std::move(box));
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
