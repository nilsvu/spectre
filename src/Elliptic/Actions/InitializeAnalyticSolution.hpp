// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic {
namespace Actions {

/*!
 * \brief Place the analytic solution of the system fields in the DataBox.
 *
 * If the `BackgroundTag` holds an `AnalyticSolutionType`, evaluate it for all
 * `AnalyticSolutionFields` and store the fields in the DataBox. Otherwise,
 * signal no analytic solution is available by storing `std::nullopt` in the
 * DataBox.
 *
 * Uses:
 * - DataBox:
 *   - `BackgroundTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `::Tags::AnalyticSolutions<AnalyticSolutionFields>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename BackgroundTag, typename AnalyticSolutionType,
          typename AnalyticSolutionFields>
struct InitializeAnalyticSolution {
  using const_global_cache_tags = tmpl::list<BackgroundTag>;
  using analytic_fields_tag = ::Tags::AnalyticSolutions<AnalyticSolutionFields>;

  using simple_tags = tmpl::list<analytic_fields_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto analytic_solution =
        dynamic_cast<const AnalyticSolutionType*>(&db::get<BackgroundTag>(box));
    if (analytic_solution != nullptr) {
      const auto& inertial_coords =
          get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
      auto analytic_fields =
          variables_from_tagged_tuple(analytic_solution->variables(
              inertial_coords, AnalyticSolutionFields{}));
      Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(analytic_fields));
    }
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace elliptic
