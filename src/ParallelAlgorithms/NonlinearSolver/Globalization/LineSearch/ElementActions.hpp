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
        make_not_null(&box), [](const gsl::not_null<double*> step_length,
                                const gsl::not_null<size_t*>
                                    globalization_iteration_id) noexcept {
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
