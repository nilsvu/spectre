// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Observe.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
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
template <typename FieldsTag>
struct UpdateHasConverged;
}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

template <typename FieldsTag, typename GlobalizationStrategy,
          typename BroadcastTarget>
struct UpdateResidualMagnitude {
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, FieldsTag>>;
  using initial_residual_magnitude_tag =
      db::add_tag_prefix<::Tags::Initial, residual_magnitude_tag>;

  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<residual_magnitude_tag,
                                              DataBox>> = nullptr>
  static void apply(DataBox& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const size_t temporal_id,
                    const size_t iteration_id,
                    const size_t globalization_iteration_id,
                    const double step_length,
                    const double residual_magnitude) noexcept {
    NonlinearSolver::observe_detail::contribute_to_reduction_observer(
        cache, temporal_id, iteration_id, globalization_iteration_id,
        step_length, residual_magnitude);

    if (UNLIKELY(iteration_id == 0)) {
      db::mutate<initial_residual_magnitude_tag>(
          make_not_null(&box), [residual_magnitude](
                                   const gsl::not_null<double*>
                                       initial_residual_magnitude) noexcept {
            *initial_residual_magnitude = residual_magnitude;
          });
    } else {
      // Make sure we are converging. Far away from the solution this is not
      // guaranteed, so we emply a globalization strategy to guide the solver
      // towards the solution when the residual doesn't decrease sufficiently.
      // The _sufficient decrease condition_ is the decrease predicted by the
      // Taylor approximation, i.e. ...
      // TODO: Add sufficient decrease condition
      if (residual_magnitude >
          (1 - get<Tags::SufficientDecreaseParameter>(box) * step_length) *
              get<residual_magnitude_tag>(box)) {
        // Do some logging
        if (UNLIKELY(static_cast<int>(get<NonlinearSolver::Tags::Verbosity>(
                         box)) >= static_cast<int>(::Verbosity::Verbose))) {
          Parallel::printf(
              "Apply globalization to decrease nonlinear solver iteration %zu "
              "residual: %e\n",
              iteration_id, residual_magnitude);
        }

        Parallel::simple_action<typename GlobalizationStrategy::perform_step>(
            Parallel::get_parallel_component<BroadcastTarget>(cache));
        return;
      }
    }

    db::mutate<residual_magnitude_tag, Tags::IterationId>(
        make_not_null(&box),
        [ residual_magnitude, iteration_id ](
            const gsl::not_null<double*> local_residual_magnitude,
            const gsl::not_null<db::item_type<Tags::IterationId>*>
                local_iteration_id) noexcept {
          *local_residual_magnitude = residual_magnitude;
          *local_iteration_id = iteration_id;
        });

    // At this point, the iteration is complete. We proceed withlogging and
    // checking convergence before broadcasting back to the elements.

    // Determine whether the nonlinear solver has converged. This invokes the
    // compute item.
    const auto& has_converged = get<NonlinearSolver::Tags::HasConverged>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(get<NonlinearSolver::Tags::Verbosity>(box)) >=
                 static_cast<int>(::Verbosity::Verbose))) {
      if (UNLIKELY(iteration_id == 0)) {
        Parallel::printf("Nonlinear solver initialized with residual %e.\n",
                         residual_magnitude);
      } else {
        Parallel::printf(
            "Nonlinear solver iteration %zu done. Remaining residual: %e\n",
            iteration_id, residual_magnitude);
      }
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(get<NonlinearSolver::Tags::Verbosity>(box)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      if (UNLIKELY(iteration_id == 0)) {
        Parallel::printf(
            "The nonlinear solver has converged without any iterations: %s\n",
            has_converged);
      } else {
        Parallel::printf(
            "The nonlinear solver has converged in %zu iterations: %s\n",
            iteration_id, has_converged);
      }
    }

    Parallel::simple_action<UpdateHasConverged<FieldsTag>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        has_converged);
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
