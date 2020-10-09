// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver::newton_raphson::detail {
template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PrepareStep;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct GlobalizeAndCompleteStep;
}  // namespace NonlinearSolver::newton_raphson::detail
/// \endcond

namespace NonlinearSolver::newton_raphson::detail {

template <typename FieldsTag, typename OptionsGroup>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_source_tag =
      db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using nonlinear_operator_applied_to_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<
                LinearSolver::Tags::IterationId<OptionsGroup>,
                LinearSolver::Tags::HasConverged<OptionsGroup>,
                nonlinear_operator_applied_to_fields_tag, correction_tag,
                NonlinearSolver::Tags::GlobalizationIterationId<OptionsGroup>,
                NonlinearSolver::Tags::StepLength<OptionsGroup>>,
            db::AddComputeTags<NonlinearSolver::Tags::ResidualCompute<
                fields_tag, nonlinear_source_tag>>>(
            std::move(box),
            // The `PrepareSolve` action populates these tags with initial
            // values
            std::numeric_limits<size_t>::max(), Convergence::HasConverged{},
            typename nonlinear_operator_applied_to_fields_tag::type{},
            typename correction_tag::type{}, std::numeric_limits<size_t>::max(),
            std::numeric_limits<double>::signaling_NaN()));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_residual_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                 static_cast<int>(::Verbosity::Debug))) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + ": Prepare solve\n",
          array_index);
    }

    // Skip the solve entirely on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>>(
            make_not_null(&box),
            [](const gsl::not_null<size_t*> iteration_id) noexcept {
              *iteration_id = 1;
            });
        // TODO: Handle immediate convergence
        constexpr size_t prepare_step_index =
            tmpl::index_of<ActionList,
                           PrepareStep<FieldsTag, OptionsGroup, Label,
                                       ArraySectionIdTag>>::value;
        return {std::move(box), false, prepare_step_index + 1};
      }
    }

    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id) noexcept {
          *iteration_id = 0;
        });

    const auto& residual = db::get<nonlinear_residual_tag>(box);
    const double local_residual_magnitude_square =
        LinearSolver::inner_product(residual, residual);
    Parallel::ReductionData<
        Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>,
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>>
        reduction_data{
            db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box), 0,
            local_residual_magnitude_square, 1.};
    if constexpr (std::is_same_v<ArraySectionIdTag, void>) {
      Parallel::contribute_to_reduction<
          CheckResidualMagnitude<FieldsTag, OptionsGroup, ParallelComponent>>(
          std::move(reduction_data),
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index],
          Parallel::get_parallel_component<
              ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));
    } else {
      Parallel::contribute_to_reduction<
          CheckResidualMagnitude<FieldsTag, OptionsGroup, ParallelComponent>,
          ParallelComponent, ArraySectionIdTag>(
          std::move(reduction_data),
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index],
          Parallel::get_parallel_component<
              ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache),
          *get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box),
          get<ArraySectionIdTag>(box));
    }

    constexpr size_t next_action_index =
        tmpl::index_of<ActionList, PrepareSolve>::value + 1;
    return {std::move(box), false, next_action_index};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct InitializeHasConverged {
  using inbox_tags = tmpl::list<Tags::GlobalizationIsComplete<OptionsGroup>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox =
        get<Tags::GlobalizationIsComplete<OptionsGroup>>(inboxes);
    return inbox.find(db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(
               box)) != inbox.end();
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Retrieve reduction data from inbox
    auto globalization_is_complete = std::move(
        tuples::get<Tags::GlobalizationIsComplete<OptionsGroup>>(inboxes)
            .extract(
                db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box))
            .mapped());
    auto& has_converged = *globalization_is_complete;

    db::mutate<LinearSolver::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                             local_has_converged) noexcept {
          *local_has_converged = std::move(has_converged);
        });

    // Skip steps entirely if the solve has already converged
    constexpr size_t complete_step_index =
        tmpl::index_of<ActionList,
                       GlobalizeAndCompleteStep<FieldsTag, OptionsGroup, Label,
                                                ArraySectionIdTag>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, InitializeHasConverged>::value;
    return {std::move(box), false,
            get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box)
                ? (complete_step_index + 1)
                : (this_action_index + 1)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PrepareStep {
 private:
  using fields_tag = FieldsTag;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using linear_operator_applied_to_correction_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, correction_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<NonlinearSolver::Tags::StepLengthReduction<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                 static_cast<int>(::Verbosity::Debug))) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + "(%zu): Prepare step\n",
          array_index,
          db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box) + 1);
    }

    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>, correction_tag,
               linear_operator_applied_to_correction_tag,
               NonlinearSolver::Tags::GlobalizationIterationId<OptionsGroup>,
               NonlinearSolver::Tags::StepLength<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id, const auto correction,
           const auto linear_operator_applied_to_correction,
           const gsl::not_null<size_t*> globalization_iteration_id,
           const gsl::not_null<double*> step_length, const auto& used_for_size,
           const size_t num_step_length_reductions) noexcept {
          ++(*iteration_id);
          // Begin the linear solve with a zero initial guess
          *correction =
              make_with_value<typename correction_tag::type>(used_for_size, 0.);
          // Since the initial guess is zero, we don't need to apply the linear
          // operator to it but can just set it to zero as well. Linear things
          // are nice...
          *linear_operator_applied_to_correction = *correction;
          // Begin line search globalization with a unity step length
          *globalization_iteration_id = 0;
          *step_length = pow(0.5, num_step_length_reductions);
        },
        db::get<fields_tag>(box),
        db::get<NonlinearSolver::Tags::StepLengthReduction<OptionsGroup>>(box));
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PerformStep {
 private:
  using fields_tag = FieldsTag;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                 static_cast<int>(::Verbosity::Debug))) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() +
              "(%zu): Perform step with length %f\n",
          array_index,
          db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
          db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box));
    }

    // Skip the solve entirely on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        constexpr size_t step_end_index = tmpl::index_of<
            ActionList, GlobalizeAndCompleteStep<FieldsTag, OptionsGroup, Label,
                                                 ArraySectionIdTag>>::value;
        return {std::move(box), false, step_end_index};
      }
    }

    // Apply the correction that the linear solve has determined to improve
    // the nonlinear solution
    db::mutate<fields_tag>(
        make_not_null(&box),
        [](const auto fields, const auto& correction,
           const double step_length) { *fields += step_length * correction; },
        db::get<correction_tag>(box),
        db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box));

    constexpr size_t next_action_index =
        tmpl::index_of<ActionList, PerformStep>::value + 1;
    return {std::move(box), false, next_action_index};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct ContributeToResidualMagnitudeReduction {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_residual_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& residual = db::get<nonlinear_residual_tag>(box);
    const double local_residual_magnitude_square =
        LinearSolver::inner_product(residual, residual);
    Parallel::ReductionData<
        Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>,
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>>
        reduction_data{
            db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
            db::get<
                NonlinearSolver::Tags::GlobalizationIterationId<OptionsGroup>>(
                box),
            local_residual_magnitude_square,
            abs(db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box))};
    if constexpr (std::is_same_v<ArraySectionIdTag, void>) {
      Parallel::contribute_to_reduction<
          CheckResidualMagnitude<FieldsTag, OptionsGroup, ParallelComponent>>(
          std::move(reduction_data),
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index],
          Parallel::get_parallel_component<
              ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));
    } else {
      Parallel::contribute_to_reduction<
          CheckResidualMagnitude<FieldsTag, OptionsGroup, ParallelComponent>,
          ParallelComponent, ArraySectionIdTag>(
          std::move(reduction_data),
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index],
          Parallel::get_parallel_component<
              ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache),
          *get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box),
          get<ArraySectionIdTag>(box));
    }
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct GlobalizeAndCompleteStep {
  using inbox_tags = tmpl::list<Tags::GlobalizationIsComplete<OptionsGroup>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox =
        get<Tags::GlobalizationIsComplete<OptionsGroup>>(inboxes);
    const auto received_data =
        inbox.find(db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    if (received_data == inbox.end()) {
      return false;
    }

    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        const bool globalization_is_complete =
            received_data->second.has_value();
        return globalization_is_complete;
      }
    }

    return true;
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Retrieve reduction data from inbox
    auto globalization_is_complete = std::move(
        tuples::get<Tags::GlobalizationIsComplete<OptionsGroup>>(inboxes)
            .extract(
                db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box))
            .mapped());

    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                 static_cast<int>(::Verbosity::Debug))) {
      if (not globalization_is_complete) {
        Parallel::printf(
            "%s " + Options::name<OptionsGroup>() + "(%zu): Globalize(%zu)\n",
            array_index,
            db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
            db::get<
                NonlinearSolver::Tags::GlobalizationIterationId<OptionsGroup>>(
                box));
      } else {
        Parallel::printf(
            "%s " + Options::name<OptionsGroup>() + "(%zu): Complete step\n",
            array_index,
            db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
      }
    }

    // Skip the solve entirely on elements that are not part of the section
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, GlobalizeAndCompleteStep>::value;
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList, PrepareStep<FieldsTag, OptionsGroup, Label,
                                               ArraySectionIdTag>>::value;
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        auto& has_converged = *globalization_is_complete;

        db::mutate<LinearSolver::Tags::HasConverged<OptionsGroup>,
                   LinearSolver::Tags::IterationId<OptionsGroup>>(
            make_not_null(&box),
            [&has_converged](
                const gsl::not_null<Convergence::HasConverged*>
                    local_has_converged,
                const gsl::not_null<size_t*> local_iteration_id) noexcept {
              *local_has_converged = std::move(has_converged);
              ++(*local_iteration_id);
            });

        return {std::move(box), false,
                get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box)
                    ? (this_action_index + 1)
                    : (prepare_step_index + 1)};
      }
    }

    if (not globalization_is_complete) {
      // Update the step length
      db::mutate<NonlinearSolver::Tags::StepLength<OptionsGroup>,
                 NonlinearSolver::Tags::GlobalizationIterationId<OptionsGroup>>(
          make_not_null(&box),
          [](const gsl::not_null<double*> step_length,
             const gsl::not_null<size_t*> globalization_iteration_id) noexcept {
            if (*globalization_iteration_id == 0) {
              *step_length *= -0.5;
            } else {
              *step_length *= 0.5;
            }
            ++(*globalization_iteration_id);
          });
      // Continue globalization by taking a step with the updated step length
      // and checking the residual again
      constexpr size_t perform_step_index =
          tmpl::index_of<ActionList, PerformStep<FieldsTag, OptionsGroup, Label,
                                                 ArraySectionIdTag>>::value;
      return {std::move(box), false, perform_step_index};
    }

    auto& has_converged = *globalization_is_complete;

    db::mutate<LinearSolver::Tags::HasConverged<OptionsGroup>,
               LinearSolver::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [&has_converged](
            const gsl::not_null<Convergence::HasConverged*> local_has_converged,
            const gsl::not_null<size_t*> local_iteration_id) noexcept {
          *local_has_converged = std::move(has_converged);
          if (*local_has_converged) {
            ++(*local_iteration_id);
          }
        });

    // Repeat steps until the solve has converged
    return {std::move(box), false,
            get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : prepare_step_index};
  }
};

}  // namespace NonlinearSolver::newton_raphson::detail
