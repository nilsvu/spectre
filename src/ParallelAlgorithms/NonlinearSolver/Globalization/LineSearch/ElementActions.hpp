// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace NonlinearSolver {
namespace Globalization {
namespace LineSearch_detail {

struct Prepare {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<NonlinearSolver::Tags::GlobalizationIterationId,
               NonlinearSolver::Tags::GlobalizationIterationsHistory,
               NonlinearSolver::Tags::StepLength,
               NonlinearSolver::Tags::GlobalizationHasConverged>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> globalization_iteration_id,
           const gsl::not_null<std::vector<size_t>*>
               globalization_iterations_history,
           const gsl::not_null<double*> step_length,
           const gsl::not_null<bool*> globalization_has_converged) noexcept {
          globalization_iterations_history->push_back(
              (*globalization_iteration_id) + 1);
          *globalization_iteration_id = 0;
          *step_length = 1.;
          *globalization_has_converged = false;
        });
    return {std::move(box)};
  }
};

struct PerformStep {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<db::tag_is_retrievable_v<
                   NonlinearSolver::Tags::GlobalizationIterationId, DataBox> and
               db::tag_is_retrievable_v<NonlinearSolver::Tags::StepLength,
                                        DataBox>> = nullptr>
  static void apply(DataBox& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) noexcept {
    db::mutate<NonlinearSolver::Tags::StepLength,
               NonlinearSolver::Tags::GlobalizationIterationId>(
        make_not_null(&box),
        [](const gsl::not_null<double*> step_length,
           const gsl::not_null<size_t*> globalization_iteration_id) noexcept {
          if (*step_length == 1.) {
            *step_length = -0.5;
          } else {
            *step_length *= 0.5;
          }
          (*globalization_iteration_id)++;
        });

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

}  // namespace LineSearch_detail
}  // namespace Globalization
}  // namespace NonlinearSolver
