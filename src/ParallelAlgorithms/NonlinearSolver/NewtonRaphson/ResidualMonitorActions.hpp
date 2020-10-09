// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace NonlinearSolver::newton_raphson::detail {

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct CheckResidualMagnitude {
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      LinearSolver::Tags::Initial<residual_magnitude_tag>;

  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<residual_magnitude_tag,
                                              DataBox>> = nullptr>
  static void apply(DataBox& box, Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const size_t iteration_id,
                    const size_t globalization_iteration_id,
                    const double residual_magnitude,
                    const double step_length) noexcept {
    if (UNLIKELY(iteration_id == 0)) {
      db::mutate<initial_residual_magnitude_tag>(
          make_not_null(&box),
          [residual_magnitude](const gsl::not_null<double*>
                                   initial_residual_magnitude) noexcept {
            *initial_residual_magnitude = residual_magnitude;
          });
    } else {
      // Make sure we are converging. Far away from the solution this is not
      // guaranteed, so we employ a globalization strategy to guide the solver
      // towards the solution when the residual doesn't decrease sufficiently.
      // The _sufficient decrease condition_ is the decrease predicted by the
      // Taylor approximation, i.e. ...
      // TODO: Add sufficient decrease condition
      const double sufficient_decrease =
          get<NonlinearSolver::Tags::SufficientDecreaseParameter<OptionsGroup>>(
              box);
      if (globalization_iteration_id < 5 and
          residual_magnitude > (1. - sufficient_decrease * step_length) *
                                   get<residual_magnitude_tag>(box)) {
        // Do some logging
        if (UNLIKELY(static_cast<int>(
                         get<LinearSolver::Tags::Verbosity<OptionsGroup>>(
                             box)) >= static_cast<int>(::Verbosity::Verbose))) {
          Parallel::printf(
              "Step with length %f didn't sufficiently decrease the '" +
                  Options::name<OptionsGroup>() +
                  "' iteration %zu residual (possible overshoot). Residual: "
                  "%e\n",
              step_length, iteration_id, residual_magnitude);
        }

        Parallel::receive_data<Tags::GlobalizationIsComplete<OptionsGroup>>(
            Parallel::get_parallel_component<BroadcastTarget>(cache),
            iteration_id, std::optional<Convergence::HasConverged>{});
        return;
      }
    }

    db::mutate<residual_magnitude_tag>(
        make_not_null(&box),
        [residual_magnitude](
            const gsl::not_null<double*> local_residual_magnitude) noexcept {
          *local_residual_magnitude = residual_magnitude;
        });

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup>(iteration_id, residual_magnitude, cache);

    // Determine whether the nonlinear solver has converged
    Convergence::HasConverged has_converged{
        get<LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>>(box),
        iteration_id, residual_magnitude,
        get<initial_residual_magnitude_tag>(box)};

    // Do some logging
    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                 static_cast<int>(::Verbosity::Quiet))) {
      if (UNLIKELY(iteration_id == 0)) {
        Parallel::printf("Nonlinear solver '" + Options::name<OptionsGroup>() +
                             "' initialized with residual %e.\n",
                         residual_magnitude);
      } else {
        Parallel::printf("Nonlinear solver '" + Options::name<OptionsGroup>() +
                             "' iteration %zu done. Remaining residual: %e\n",
                         iteration_id, residual_magnitude);
      }
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      if (UNLIKELY(iteration_id == 0)) {
        Parallel::printf("The nonlinear solver '" +
                             Options::name<OptionsGroup>() +
                             "' has converged without any iterations: %s\n",
                         has_converged);
      } else {
        Parallel::printf("The nonlinear solver '" +
                             Options::name<OptionsGroup>() +
                             "' has converged in %zu iterations: %s\n",
                         iteration_id, has_converged);
      }
    }

    Parallel::receive_data<Tags::GlobalizationIsComplete<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        std::optional<Convergence::HasConverged>(std::move(has_converged)));
  }
};

}  // namespace NonlinearSolver::newton_raphson::detail
