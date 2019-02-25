// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/Initialization/BoundaryConditions.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitorActions.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver {
namespace newton_raphson_detail {
template <typename Metavariables>
struct ResidualMonitor;
}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

struct InitializeHasConverged {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const db::item_type<NonlinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<NonlinearSolver::Tags::HasConverged>(
        make_not_null(&box), [&has_converged](
                                 const gsl::not_null<db::item_type<
                                     NonlinearSolver::Tags::HasConverged>*>
                                     local_has_converged) noexcept {
          *local_has_converged = has_converged;
        });
  }
};

struct PerformStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*mete*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using fields_tag = typename Metavariables::system::nonlinear_fields_tag;
    using correction_tag =
        db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;

    // Keep iterating the linear solver while it has not yet converged
    if (not get<LinearSolver::Tags::HasConverged>(box)) {
      return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), false);
    }

    // Apply the correction that the linear solve has determined to improve
    // the nonlinear solution
    db::mutate<fields_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<fields_tag>*> fields,
           const db::item_type<correction_tag>& correction) {
          *fields += correction;
        },
        get<correction_tag>(box));

    return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), false);
  }
};

struct PrepareLinearSolve {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*mete*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using fields_tag = typename Metavariables::system::nonlinear_fields_tag;
    using nonlinear_source_tag = db::add_tag_prefix<::Tags::Source, fields_tag>;
    using nonlinear_operator_tag =
        db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo,
                           fields_tag>;
    using correction_tag =
        db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
    using linear_source_tag =
        db::add_tag_prefix<::Tags::Source, correction_tag>;
    using linear_operator_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                           correction_tag>;

    // Keep iterating the linear solver while it has not yet converged
    if (not get<LinearSolver::Tags::HasConverged>(box)) {
      return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), false);
    }

    db::mutate<linear_source_tag, correction_tag, linear_operator_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<linear_source_tag>*> linear_source,
           const gsl::not_null<db::item_type<correction_tag>*> correction,
           const gsl::not_null<db::item_type<linear_operator_tag>*>
               linear_operator,
           const db::item_type<nonlinear_source_tag>& nonlinear_source,
           const db::item_type<nonlinear_operator_tag>& nonlinear_operator) {
          // Compute new nonlinear residual, which sources the next linear
          // solve. The nonlinear source b stays the same, but need to apply the
          // nonlinear operator to the new field values. We assume this has been
          // done at this point.
          *linear_source = nonlinear_source - nonlinear_operator;
          // Begin the linear solve with a zero initial guess
          // TODO: Why does this not work? correction has small deviation that
          // mimic linear_source (but linear_operator below is correct)
          *correction = make_with_value<db::item_type<correction_tag>>(
              nonlinear_source, 0.);
          *linear_operator = *correction;
        },
        get<nonlinear_source_tag>(box), get<nonlinear_operator_tag>(box));

    Parallel::contribute_to_reduction<UpdateResidual<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            LinearSolver::inner_product(get<linear_source_tag>(box),
                                        get<linear_source_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    // Terminate algorithm for now. The reduction will be broadcast to the
    // next action which is responsible for restarting the algorithm.
    return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), true);
  }
};

struct UpdateHasConverged {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const db::item_type<NonlinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<NonlinearSolver::Tags::HasConverged,
               NonlinearSolver::Tags::IterationId>(
        make_not_null(&box), [&has_converged](
                                 const gsl::not_null<db::item_type<
                                     NonlinearSolver::Tags::HasConverged>*>
                                     local_has_converged,
                                 const gsl::not_null<db::item_type<
                                     NonlinearSolver::Tags::IterationId>*>
                                     iteration_id) noexcept {
          *local_has_converged = has_converged;
          // Prepare for next iteration
          (*iteration_id)++;
        });

    // Proceed with algorithm
    // We use `ckLocal()` here since this is essentially retrieving "self",
    // which is guaranteed to be on the local processor. This ensures the calls
    // are evaluated in order.
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .ckLocal()
        ->set_terminate(false);
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm();
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
