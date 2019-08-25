// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace NonlinearSolver {
namespace Globalization {
namespace LineSearch_detail {

struct InitializeElement {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<
                NonlinearSolver::Tags::GlobalizationIterationId,
                NonlinearSolver::Tags::GlobalizationIterationsHistory,
                NonlinearSolver::Tags::StepLength,
                NonlinearSolver::Tags::GlobalizationHasConverged>,
            db::AddComputeTags<NonlinearSolver::Tags::TemporalIdCompute,
                               NonlinearSolver::Tags::TemporalIdNextCompute>>(
            std::move(box), size_t{0}, std::vector<size_t>{}, 1., false));
  }
};

struct ReinitializeElement {
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
        make_not_null(&box), [](const gsl::not_null<size_t*>
                                    globalization_iteration_id,
                                const gsl::not_null<std::vector<size_t>*>
                                    globalization_iterations_history,
                                const gsl::not_null<double*> step_length,
                                const gsl::not_null<bool*>
                                    globalization_has_converged) noexcept {
          globalization_iterations_history->push_back(
              (*globalization_iteration_id) + 1);
          *globalization_iteration_id = 0;
          *step_length = 1.;
          *globalization_has_converged = false;
        });
    return {std::move(box)};
  }
};

}  // namespace LineSearch_detail
}  // namespace Globalization
}  // namespace NonlinearSolver
