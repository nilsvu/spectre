// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

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
    using compute_tags =
        tmpl::list<NonlinearSolver::Tags::TemporalIdCompute,
                   NonlinearSolver::Tags::TemporalIdNextCompute>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<
                NonlinearSolver::Tags::GlobalizationIterationId,
                NonlinearSolver::Tags::GlobalizationIterationsHistory,
                NonlinearSolver::Tags::StepLength,
                NonlinearSolver::Tags::GlobalizationHasConverged>,
            compute_tags>(std::move(box), std::numeric_limits<size_t>::max(),
                          std::vector<size_t>{}, 1., false));
  }
};

}  // namespace LineSearch_detail
}  // namespace Globalization
}  // namespace NonlinearSolver
