// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GetSection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver::gmres::detail {
template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor;
template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct PrepareStep;
template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct NormalizeOperandAndUpdateField;
template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct CompleteStep;
}  // namespace LinearSolver::gmres::detail
/// \endcond

namespace LinearSolver::gmres::detail {

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename SourceTag, typename ArraySectionIdTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using initial_fields_tag = db::add_tag_prefix<::Tags::Initial, fields_tag>;
  using source_tag = SourceTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>>(
        [](const gsl::not_null<size_t*> iteration_id) { *iteration_id = 0; },
        make_not_null(&box));

    // Skip the initial reduction on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }
    }

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s: Prepare solve\n", get_output(array_index),
                       pretty_type::name<OptionsGroup>());
    }

    db::mutate<operand_tag, initial_fields_tag, basis_history_tag>(
        [](const auto operand, const auto initial_fields,
           const auto basis_history, const auto& source,
           const auto& operator_applied_to_fields, const auto& fields) {
          *operand = source - operator_applied_to_fields;
          *initial_fields = fields;
          *basis_history = typename basis_history_tag::type{};
        },
        make_not_null(&box), get<source_tag>(box),
        get<operator_applied_to_fields_tag>(box), get<fields_tag>(box));

    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<InitializeResidualMagnitude<
        FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            inner_product(get<operand_tag>(box), get<operand_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache),
        make_not_null(&section));

    if constexpr (Preconditioned) {
      using preconditioned_operand_tag =
          db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
      using preconditioned_basis_history_tag =
          LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

      db::mutate<preconditioned_basis_history_tag>(
          [](const auto preconditioned_basis_history) {
            *preconditioned_basis_history =
                typename preconditioned_basis_history_tag::type{};
          },
          make_not_null(&box));
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct NormalizeInitialOperand {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;
  using inbox_tags = tmpl::list<Tags::InitialOrthogonalization<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    auto& inbox = get<Tags::InitialOrthogonalization<OptionsGroup>>(inboxes);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    auto received_data = std::move(inbox.extract(iteration_id).mapped());
    const double residual_magnitude = get<0>(received_data);
    auto& has_converged = get<1>(received_data);
    db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
        [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                             local_has_converged) {
          *local_has_converged = std::move(has_converged);
        },
        make_not_null(&box));

    // Skip steps entirely if the solve has already converged
    constexpr size_t step_end_index =
        tmpl::index_of<ActionList,
                       CompleteStep<FieldsTag, OptionsGroup, Preconditioned,
                                    Label, ArraySectionIdTag>>::value;
    if (get<Convergence::Tags::HasConverged<OptionsGroup>>(box)) {
      return {Parallel::AlgorithmExecution::Continue, step_end_index + 1};
    }

    // Skip the solve entirely on elements that are not part of the section. To
    // do so, we skip ahead to the `ApplyOperatorActions` action list between
    // `PrepareStep` and `PerformStep`. Those actions need a chance to run even
    // on elements that are not part of the section, because they may take part
    // in preconditioning (see Multigrid preconditioner).
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }
    }

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Normalize initial operand\n",
                       get_output(array_index),
                       pretty_type::name<OptionsGroup>(), iteration_id);
    }

    db::mutate<operand_tag, basis_history_tag>(
        [residual_magnitude](const auto operand, const auto basis_history) {
          *operand /= residual_magnitude;
          basis_history->push_back(*operand);
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct PrepareStep {
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>>(
        [](const gsl::not_null<size_t*> iteration_id) { ++(*iteration_id); },
        make_not_null(&box));

    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }
    }

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s %s(%zu): Prepare step\n", get_output(array_index),
          pretty_type::name<OptionsGroup>(),
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box));
    }

    if constexpr (Preconditioned) {
      using fields_tag = FieldsTag;
      using operand_tag =
          db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
      using preconditioned_operand_tag =
          db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
      using operator_tag = db::add_tag_prefix<
          LinearSolver::Tags::OperatorAppliedTo,
          std::conditional_t<Preconditioned, preconditioned_operand_tag,
                             operand_tag>>;

      db::mutate<preconditioned_operand_tag, operator_tag>(
          [](const auto preconditioned_operand,
             const auto operator_applied_to_operand, const auto& operand) {
            // Start the preconditioner at zero because we have no reason to
            // expect the remaining residual to have a particular form.
            // Another possibility would be to start the preconditioner with an
            // initial guess equal to its source, so not running the
            // preconditioner at all means it is the identity, but that approach
            // appears to yield worse results.
            *preconditioned_operand =
                make_with_value<typename preconditioned_operand_tag::type>(
                    operand, 0.);
            // Also set the operator applied to the initial preconditioned
            // operand to zero because it's linear. This may save the
            // preconditioner an operator application if it's optimized for
            // this.
            *operator_applied_to_operand =
                make_with_value<typename operator_tag::type>(operand, 0.);
          },
          make_not_null(&box), get<operand_tag>(box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct PerformStep {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using preconditioned_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Skip to the end of the step on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        constexpr size_t step_end_index =
            tmpl::index_of<ActionList,
                           NormalizeOperandAndUpdateField<
                               FieldsTag, OptionsGroup, Preconditioned, Label,
                               ArraySectionIdTag>>::value;
        return {Parallel::AlgorithmExecution::Continue, step_end_index};
      }
    }

    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Perform step\n", get_output(array_index),
                       pretty_type::name<OptionsGroup>(), iteration_id);
    }

    using operator_tag = db::add_tag_prefix<
        LinearSolver::Tags::OperatorAppliedTo,
        std::conditional_t<Preconditioned, preconditioned_operand_tag,
                           operand_tag>>;
    using orthogonalization_iteration_id_tag =
        LinearSolver::Tags::Orthogonalization<
            Convergence::Tags::IterationId<OptionsGroup>>;
    using basis_history_tag =
        LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

    if constexpr (Preconditioned) {
      using preconditioned_basis_history_tag =
          LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

      db::mutate<preconditioned_basis_history_tag>(
          [](const auto preconditioned_basis_history,
             const auto& preconditioned_operand) {
            preconditioned_basis_history->push_back(preconditioned_operand);
          },
          make_not_null(&box), get<preconditioned_operand_tag>(box));
    }

    db::mutate<operand_tag, orthogonalization_iteration_id_tag>(
        [](const auto operand,
           const gsl::not_null<size_t*> orthogonalization_iteration_id,
           const auto& operator_action) {
          *operand = typename operand_tag::type(operator_action);
          *orthogonalization_iteration_id = 0;
        },
        make_not_null(&box), get<operator_tag>(box));

    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<
        StoreOrthogonalization<FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            get<Convergence::Tags::IterationId<OptionsGroup>>(box),
            get<orthogonalization_iteration_id_tag>(box),
            inner_product(get<basis_history_tag>(box)[0],
                          get<operand_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache),
        make_not_null(&section));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct OrthogonalizeOperand {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using orthogonalization_iteration_id_tag =
      LinearSolver::Tags::Orthogonalization<
          Convergence::Tags::IterationId<OptionsGroup>>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  using inbox_tags = tmpl::list<Tags::Orthogonalization<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    auto& inbox = get<Tags::Orthogonalization<OptionsGroup>>(inboxes);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    const double orthogonalization =
        std::move(inbox.extract(iteration_id).mapped());

    db::mutate<operand_tag, orthogonalization_iteration_id_tag>(
        [orthogonalization](
            const auto operand,
            const gsl::not_null<size_t*> orthogonalization_iteration_id,
            const auto& basis_history) {
          *operand -= orthogonalization *
                      gsl::at(basis_history, *orthogonalization_iteration_id);
          ++(*orthogonalization_iteration_id);
        },
        make_not_null(&box), get<basis_history_tag>(box));

    const auto& next_orthogonalization_iteration_id =
        get<orthogonalization_iteration_id_tag>(box);
    const bool orthogonalization_complete =
        next_orthogonalization_iteration_id == iteration_id;
    const double local_orthogonalization =
        inner_product(orthogonalization_complete
                          ? get<operand_tag>(box)
                          : gsl::at(get<basis_history_tag>(box),
                                    next_orthogonalization_iteration_id),
                      get<operand_tag>(box));

    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<
        StoreOrthogonalization<FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            iteration_id, next_orthogonalization_iteration_id,
            local_orthogonalization},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache),
        make_not_null(&section));

    // Repeat this action until orthogonalization is complete
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, OrthogonalizeOperand>::value;
    return {Parallel::AlgorithmExecution::Continue,
            orthogonalization_complete ? (this_action_index + 1)
                                       : this_action_index};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct NormalizeOperandAndUpdateField {
 private:
  using fields_tag = FieldsTag;
  using initial_fields_tag = db::add_tag_prefix<::Tags::Initial, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using preconditioned_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;
  using preconditioned_basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<std::conditional_t<
          Preconditioned, preconditioned_operand_tag, operand_tag>>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;
  using inbox_tags = tmpl::list<Tags::FinalOrthogonalization<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    auto& inbox = get<Tags::FinalOrthogonalization<OptionsGroup>>(inboxes);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    // Retrieve reduction data from inbox
    auto received_data = std::move(inbox.extract(iteration_id).mapped());
    const double normalization = get<0>(received_data);
    const auto& minres = get<1>(received_data);
    db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
        [&received_data](
            const gsl::not_null<Convergence::HasConverged*> has_converged) {
          *has_converged = std::move(get<2>(received_data));
        },
        make_not_null(&box));

    // Elements that are not part of the section jump directly to the
    // `ApplyOperationActions` for the next step.
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      constexpr size_t complete_step_index =
          tmpl::index_of<ActionList,
                         CompleteStep<FieldsTag, OptionsGroup, Preconditioned,
                                      Label, ArraySectionIdTag>>::value;
      constexpr size_t prepare_step_index =
          tmpl::index_of<ActionList,
                         PrepareStep<FieldsTag, OptionsGroup, Preconditioned,
                                     Label, ArraySectionIdTag>>::value;
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        return {Parallel::AlgorithmExecution::Continue,
                get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                    ? (complete_step_index + 1)
                    : prepare_step_index};
      }
    }

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Update field\n", get_output(array_index),
                       pretty_type::name<OptionsGroup>(), iteration_id);
    }

    db::mutate<operand_tag, basis_history_tag, fields_tag>(
        [normalization, &minres](const auto operand, const auto basis_history,
                                 const auto field, const auto& initial_field,
                                 const auto& preconditioned_basis_history,
                                 const auto& has_converged) {
          // Avoid an FPE if the new operand norm is exactly zero. In that case
          // the problem is solved and the algorithm will terminate (see
          // Proposition 9.3 in \cite Saad2003). Since there will be no next
          // iteration we don't need to normalize the operand.
          if (LIKELY(normalization > 0.)) {
            *operand /= normalization;
          }
          basis_history->push_back(*operand);
          // Don't update the solution if an error occurred
          if (not(has_converged and
                  has_converged.reason() == Convergence::Reason::Error)) {
            *field = initial_field;
            for (size_t i = 0; i < minres.size(); i++) {
              *field += minres[i] * gsl::at(preconditioned_basis_history, i);
            }
          }
        },
        make_not_null(&box), get<initial_fields_tag>(box),
        get<preconditioned_basis_history_tag>(box),
        get<Convergence::Tags::HasConverged<OptionsGroup>>(box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// Jump back to `PrepareStep` to continue iterating if the algorithm has not yet
// converged, or complete the solve and proceed with the action list if it has
// converged. This is a separate action because the user has the opportunity to
// insert actions before the step completes, for example to do observations.
template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename ArraySectionIdTag>
struct CompleteStep {
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s %s(%zu): Complete step\n", get_output(array_index),
          pretty_type::name<OptionsGroup>(),
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box));
    }

    // Repeat steps until the solve has converged
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList,
                       PrepareStep<FieldsTag, OptionsGroup, Preconditioned,
                                   Label, ArraySectionIdTag>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, CompleteStep>::value;
    return {Parallel::AlgorithmExecution::Continue,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : prepare_step_index};
  }
};

}  // namespace LinearSolver::gmres::detail
