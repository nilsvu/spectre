// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using initial_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using preconditioned_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
  using operator_applied_to_operand_tag = db::add_tag_prefix<
      LinearSolver::Tags::OperatorAppliedTo,
      std::conditional_t<Preconditioned, preconditioned_operand_tag,
                         operand_tag>>;
  using orthogonalization_iteration_id_tag =
      db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                         LinearSolver::Tags::IterationId<OptionsGroup>>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto initial_box = ::Initialization::merge_into_databox<
        InitializeElement,
        db::AddSimpleTags<LinearSolver::Tags::IterationId<OptionsGroup>,
                          initial_fields_tag, operator_applied_to_fields_tag,
                          operand_tag, preconditioned_operand_tag,
                          operator_applied_to_operand_tag,
                          orthogonalization_iteration_id_tag, basis_history_tag,
                          LinearSolver::Tags::HasConverged<OptionsGroup>>>(
        std::move(box),
        // The `PrepareSolve` action populates these tags with initial values,
        // except for `operator_applied_to_fields_tag` which is expected to be
        // filled at that point and `operator_applied_to_operand_tag` which is
        // expected to be updated in every iteration of the algorithm.
        std::numeric_limits<size_t>::max(), db::item_type<initial_fields_tag>{},
        db::item_type<operator_applied_to_fields_tag>{},
        db::item_type<operand_tag>{},
        db::item_type<preconditioned_operand_tag>{},
        db::item_type<operator_applied_to_operand_tag>{},
        std::numeric_limits<size_t>::max(), db::item_type<basis_history_tag>{},
        Convergence::HasConverged{});

    auto preconditioning_box = initialize_preconditioning(
        std::move(initial_box), std::bool_constant<Preconditioned>{});

    return std::make_tuple(std::move(preconditioning_box));
  }

 private:
  template <typename DbTagsList>
  static auto initialize_preconditioning(
      db::DataBox<DbTagsList>&& box,
      std::true_type /* preconditioning_enabled */) noexcept {
    using preconditioned_basis_history_tag =
        LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

    return ::Initialization::merge_into_databox<
        InitializeElement, db::AddSimpleTags<preconditioned_basis_history_tag>>(
        std::move(box), db::item_type<preconditioned_basis_history_tag>{});
  }

  template <typename DbTagsList>
  static db::DataBox<DbTagsList> initialize_preconditioning(
      db::DataBox<DbTagsList>&& box,
      std::false_type /* preconditioning_disabled */) noexcept {
    return std::move(box);
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
