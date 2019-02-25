// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Observe.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver {
namespace newton_raphson_detail {
struct InitializeHasConverged;
struct UpdateHasConverged;
}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

template <typename BroadcastTarget>
struct InitializeResidual {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double residual_magnitude) noexcept {
    using fields_tag = typename Metavariables::system::nonlinear_fields_tag;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>;
    using initial_residual_magnitude_tag =
        db::add_tag_prefix<::Tags::Initial, residual_magnitude_tag>;

    db::mutate<residual_magnitude_tag, initial_residual_magnitude_tag>(
        make_not_null(&box),
        [residual_magnitude](
            const gsl::not_null<double*> local_residual_magnitude,
            const gsl::not_null<double*>
                local_initial_residual_magnitude) noexcept {
          *local_residual_magnitude = *local_initial_residual_magnitude =
              residual_magnitude;
        });

    NonlinearSolver::observe_detail::contribute_to_reduction_observer(box,
                                                                      cache);

    // Determine whether the nonlinear solver has converged. This invokes the
    // compute item.
    const auto& has_converged =
        db::get<NonlinearSolver::Tags::HasConverged>(box);

    if (UNLIKELY(static_cast<int>(get<NonlinearSolver::OptionTags::Verbosity>(
                     cache)) >= static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf("Nonlinear solver initialized with residual %e.\n",
                       residual_magnitude);
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(get<NonlinearSolver::OptionTags::Verbosity>(
                     cache)) >= static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf(
          "The nonlinear solver has converged without any iterations: %s",
          has_converged);
    }

    Parallel::simple_action<InitializeHasConverged>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        has_converged);
  }
};

template <typename BroadcastTarget>
struct UpdateResidual {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double residual_magnitude) noexcept {
    using fields_tag = typename Metavariables::system::nonlinear_fields_tag;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>;

    db::mutate<residual_magnitude_tag, NonlinearSolver::Tags::IterationId>(
        make_not_null(&box), [residual_magnitude](
                                 const gsl::not_null<double*>
                                     local_residual_magnitude,
                                 const gsl::not_null<db::item_type<
                                     NonlinearSolver::Tags::IterationId>*>
                                     iteration_id) noexcept {
          *local_residual_magnitude = residual_magnitude;
          // Prepare for the next iteration
          (*iteration_id)++;
        });

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    NonlinearSolver::observe_detail::contribute_to_reduction_observer(box,
                                                                      cache);

    // Determine whether the nonlinear solver has converged. This invokes the
    // compute item.
    const auto& has_converged = get<NonlinearSolver::Tags::HasConverged>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(get<NonlinearSolver::OptionTags::Verbosity>(
                     cache)) >= static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf(
          "Nonlinear solver iteration %zu done. Remaining residual: %e\n",
          get<NonlinearSolver::Tags::IterationId>(box),
          get<residual_magnitude_tag>(box));
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(get<NonlinearSolver::OptionTags::Verbosity>(
                     cache)) >= static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf(
          "The nonlinear solver has converged in %zu iterations: %s",
          get<NonlinearSolver::Tags::IterationId>(box), has_converged);
    }

    Parallel::simple_action<UpdateHasConverged>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        has_converged);
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
