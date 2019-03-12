// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace cg_detail {
template <typename Metavariables>
struct ResidualMonitor;
template <typename BroadcastTarget, bool IsReinitializing>
struct InitializeResidual;
}  // namespace cg_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace cg_detail {

template <typename Metavariables>
struct InitializeElement {
 private:
  using fields_tag = typename Metavariables::system::fields_tag;
  using source_tag = db::add_tag_prefix<::Tags::Source, fields_tag>;
  using operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  using simple_tags =
      db::AddSimpleTags<LinearSolver::Tags::IterationId,
                        ::Tags::Next<LinearSolver::Tags::IterationId>,
                        residual_tag, LinearSolver::Tags::HasConverged>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList, typename ArrayIndex, typename ParallelComponent>
  static auto initialize(db::DataBox<TagsList>&& box,
                         const Parallel::ConstGlobalCache<Metavariables>& cache,
                         const ArrayIndex& array_index,
                         const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<operand_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<operand_tag>*> p,
           const db::item_type<source_tag>& b,
           const db::item_type<operator_tag>& Ax) noexcept { *p = b - Ax; },
        get<source_tag>(box), get<operator_tag>(box));
    auto r = db::item_type<residual_tag>(get<operand_tag>(box));

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    Parallel::contribute_to_reduction<
        cg_detail::InitializeResidual<ParallelComponent, false>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            inner_product(r, r)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), db::item_type<LinearSolver::Tags::IterationId>{0},
        db::item_type<::Tags::Next<LinearSolver::Tags::IterationId>>{1},
        std::move(r), db::item_type<LinearSolver::Tags::HasConverged>{});
  }

  // This function is called as a simple action for re-initialization.
  // When we do initialization in a phase-dependent action list, both functions
  // in this struct will be made into iterative actions.
  template <typename... DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearSolver::Tags::IterationId,
               ::Tags::Next<LinearSolver::Tags::IterationId>, operand_tag,
               residual_tag, LinearSolver::Tags::HasConverged>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<LinearSolver::Tags::IterationId>*>
               iteration_id,
           const gsl::not_null<
               db::item_type<::Tags::Next<LinearSolver::Tags::IterationId>>*>
               next_iteration_id,
           const gsl::not_null<db::item_type<operand_tag>*> p,
           const gsl::not_null<db::item_type<residual_tag>*> r,
           const gsl::not_null<db::item_type<LinearSolver::Tags::HasConverged>*>
               has_converged,
           const db::item_type<source_tag>& b,
           const db::item_type<operator_tag>& Ax) noexcept {
          *iteration_id = 0;
          *next_iteration_id = 1;
          *p = b - Ax;
          *r = *p;
          *has_converged = db::item_type<LinearSolver::Tags::HasConverged>{};
        },
        get<source_tag>(box), get<operator_tag>(box));

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    Parallel::contribute_to_reduction<
        cg_detail::InitializeResidual<ParallelComponent, true>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            inner_product(get<residual_tag>(box), get<residual_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
