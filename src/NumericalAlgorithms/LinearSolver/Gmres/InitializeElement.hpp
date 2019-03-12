// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/MakeWithValue.hpp"

#include "Parallel/Printf.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace gmres_detail {
template <typename Metavariables>
struct ResidualMonitor;
template <typename BroadcastTarget, bool IsReinitializing>
struct InitializeResidualMagnitude;
}  // namespace gmres_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

template <typename Metavariables>
struct InitializeElement {
 private:
  using fields_tag = typename Metavariables::system::fields_tag;
  using source_tag = db::add_tag_prefix<::Tags::Source, fields_tag>;
  using operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using initial_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using orthogonalization_iteration_id_tag =
      db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                         LinearSolver::Tags::IterationId>;
  using basis_history_tag = LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>;

 public:
  using simple_tags =
      db::AddSimpleTags<initial_fields_tag, orthogonalization_iteration_id_tag,
                        basis_history_tag, LinearSolver::Tags::HasConverged>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList, typename ArrayIndex, typename ParallelComponent>
  static auto initialize(db::DataBox<TagsList>&& box,
                         const Parallel::ConstGlobalCache<Metavariables>& cache,
                         const ArrayIndex& array_index,
                         const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<operand_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<operand_tag>*> q,
           const db::item_type<source_tag>& b,
           const db::item_type<operator_tag>& Ax) noexcept { *q = b - Ax; },
        get<source_tag>(box), get<operator_tag>(box));
    const auto& q = get<operand_tag>(box);

    Parallel::contribute_to_reduction<
        gmres_detail::InitializeResidualMagnitude<ParallelComponent, false>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            inner_product(q, q)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    db::item_type<initial_fields_tag> x0(get<fields_tag>(box));
    db::item_type<basis_history_tag> basis_history{};

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(x0),
        db::item_type<orthogonalization_iteration_id_tag>{0},
        std::move(basis_history),
        db::item_type<LinearSolver::Tags::HasConverged>{});
  }

  // This function is called as a simple action for re-initialization.
  // When we do initialization in a phase-dependent action list, both functions
  // in this struct will be made into iterative actions.
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<operand_tag, LinearSolver::Tags::IterationId, initial_fields_tag,
               orthogonalization_iteration_id_tag, basis_history_tag,
               LinearSolver::Tags::HasConverged>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<operand_tag>*> q,
           const gsl::not_null<db::item_type<LinearSolver::Tags::IterationId>*>
               iteration_id,
           const gsl::not_null<db::item_type<initial_fields_tag>*>
               initial_fields,
           const gsl::not_null<
               db::item_type<orthogonalization_iteration_id_tag>*>
               orthogonalization_iteration_id,
           const gsl::not_null<db::item_type<basis_history_tag>*> basis_history,
           const gsl::not_null<db::item_type<LinearSolver::Tags::HasConverged>*>
               has_converged,
           const db::item_type<fields_tag>& x0,
           const db::item_type<source_tag>& b,
           const db::item_type<operator_tag>& Ax) noexcept {
          *q = b - Ax;
          *iteration_id = 0;
          *initial_fields = db::item_type<initial_fields_tag>(x0);
          *orthogonalization_iteration_id = 0;
          *basis_history = db::item_type<basis_history_tag>{};
          *has_converged = db::item_type<LinearSolver::Tags::HasConverged>{};
        },
        get<fields_tag>(box), get<source_tag>(box), get<operator_tag>(box));
    const auto& q = get<operand_tag>(box);

    Parallel::contribute_to_reduction<
        gmres_detail::InitializeResidualMagnitude<ParallelComponent, true>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            inner_product(q, q)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), true);
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
