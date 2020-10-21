// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace NonlinearSolver::newton_raphson::detail {
template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PrepareStep;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct Globalize;
}  // namespace NonlinearSolver::newton_raphson::detail
/// \endcond

namespace NonlinearSolver::newton_raphson::detail {

using ResidualReductionData = Parallel::ReductionData<
    // Iteration ID
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Globalization iteration ID
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual magnitude square
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // Step length
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using nonlinear_operator_applied_to_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using globalization_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Globalization, fields_tag>;

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
            db::AddSimpleTags<Convergence::Tags::IterationId<OptionsGroup>,
                              Convergence::Tags::HasConverged<OptionsGroup>,
                              nonlinear_operator_applied_to_fields_tag,
                              correction_tag,
                              NonlinearSolver::Tags::Globalization<
                                  Convergence::Tags::IterationId<OptionsGroup>>,
                              NonlinearSolver::Tags::StepLength<OptionsGroup>,
                              globalization_fields_tag>,
            db::AddComputeTags<NonlinearSolver::Tags::ResidualCompute<
                fields_tag, source_tag>>>(
            std::move(box), std::numeric_limits<size_t>::max(),
            Convergence::HasConverged{},
            typename nonlinear_operator_applied_to_fields_tag::type{},
            typename correction_tag::type{}, std::numeric_limits<size_t>::max(),
            std::numeric_limits<double>::signaling_NaN(),
            typename globalization_fields_tag::type{}));
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
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + ": Prepare solve\n",
          array_index);
    }

    // Skip the solve entirely on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        db::mutate<Convergence::Tags::IterationId<OptionsGroup>>(
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

    db::mutate<Convergence::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id) noexcept {
          *iteration_id = 0;
        });

    // Perform a global reduction to compute the initial residual magnitude
    const auto& residual = db::get<nonlinear_residual_tag>(box);
    const double local_residual_magnitude_square =
        LinearSolver::inner_product(residual, residual);
    ResidualReductionData reduction_data{0, 0, local_residual_magnitude_square,
                                         1.};
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

    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, PrepareSolve>::value;
    return {std::move(box), false, this_action_index + 1};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct InitializeHasConverged {
  using inbox_tags = tmpl::list<Tags::GlobalizationResult<OptionsGroup>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox = get<Tags::GlobalizationResult<OptionsGroup>>(inboxes);
    return inbox.find(db::get<Convergence::Tags::IterationId<OptionsGroup>>(
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
    auto globalization_result = std::move(
        tuples::get<Tags::GlobalizationResult<OptionsGroup>>(inboxes)
            .extract(db::get<Convergence::Tags::IterationId<OptionsGroup>>(box))
            .mapped());
    ASSERT(
        std::holds_alternative<Convergence::HasConverged>(globalization_result),
        "No globalization should occur for the initial residual");
    auto& has_converged = get<Convergence::HasConverged>(globalization_result);

    db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                             local_has_converged) noexcept {
          *local_has_converged = std::move(has_converged);
        });

    // Skip steps entirely if the solve has already converged
    constexpr size_t complete_step_index =
        tmpl::index_of<ActionList, Globalize<FieldsTag, OptionsGroup, Label,
                                             ArraySectionIdTag>>::value +
        1;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, InitializeHasConverged>::value;
    return {std::move(box), false,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? complete_step_index
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
  using globalization_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Globalization, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<NonlinearSolver::Tags::DampingFactor<OptionsGroup>,
                 logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + "(%zu): Prepare step\n",
          array_index,
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box) + 1);
    }

    db::mutate<Convergence::Tags::IterationId<OptionsGroup>, correction_tag,
               linear_operator_applied_to_correction_tag,
               NonlinearSolver::Tags::Globalization<
                   Convergence::Tags::IterationId<OptionsGroup>>,
               NonlinearSolver::Tags::StepLength<OptionsGroup>,
               globalization_fields_tag>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id, const auto correction,
           const auto linear_operator_applied_to_correction,
           const gsl::not_null<size_t*> globalization_iteration_id,
           const gsl::not_null<double*> step_length,
           const auto globalization_fields, const auto& fields,
           const double damping_factor) noexcept {
          ++(*iteration_id);
          // Begin the linear solve with a zero initial guess
          *correction =
              make_with_value<typename correction_tag::type>(fields, 0.);
          // Since the initial guess is zero, we don't need to apply the linear
          // operator to it but can just set it to zero as well. Linear things
          // are nice :)
          *linear_operator_applied_to_correction = make_with_value<
              typename linear_operator_applied_to_correction_tag::type>(fields,
                                                                        0.);
          // Prepare line search globalization
          *globalization_iteration_id = 0;
          *step_length = damping_factor;
          *globalization_fields = fields;
        },
        db::get<fields_tag>(box),
        db::get<NonlinearSolver::Tags::DampingFactor<OptionsGroup>>(box));
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
  using globalization_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Globalization, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() +
              "(%zu): Perform step with length %f\n",
          array_index,
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box),
          db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box));
    }

    // Skip the solve entirely on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        constexpr size_t globalize_index =
            tmpl::index_of<ActionList, Globalize<FieldsTag, OptionsGroup, Label,
                                                 ArraySectionIdTag>>::value;
        return {std::move(box), false, globalize_index};
      }
    }

    // Apply the correction that the linear solve has determined to attempt
    // improving the nonlinear solution
    db::mutate<fields_tag>(
        make_not_null(&box),
        [](const auto fields, const auto& correction, const double step_length,
           const auto& globalization_fields) {
          *fields = globalization_fields + step_length * correction;
        },
        db::get<correction_tag>(box),
        db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box),
        db::get<globalization_fields_tag>(box));

    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, PerformStep>::value;
    return {std::move(box), false, this_action_index + 1};
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
    ResidualReductionData reduction_data{
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box),
        db::get<NonlinearSolver::Tags::Globalization<
            Convergence::Tags::IterationId<OptionsGroup>>>(box),
        local_residual_magnitude_square,
        db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box)};
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
struct Globalize {
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;
  using inbox_tags = tmpl::list<Tags::GlobalizationResult<OptionsGroup>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox = get<Tags::GlobalizationResult<OptionsGroup>>(inboxes);
    const auto received_data =
        inbox.find(db::get<Convergence::Tags::IterationId<OptionsGroup>>(box));
    if (received_data == inbox.end()) {
      return false;
    }
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        const bool globalization_is_complete =
            std::holds_alternative<Convergence::HasConverged>(
                received_data->second);
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
    auto globalization_result = std::move(
        tuples::get<Tags::GlobalizationResult<OptionsGroup>>(inboxes)
            .extract(db::get<Convergence::Tags::IterationId<OptionsGroup>>(box))
            .mapped());

    // Skip the solve entirely on elements that are not part of the section
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, Globalize>::value;
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList, PrepareStep<FieldsTag, OptionsGroup, Label,
                                               ArraySectionIdTag>>::value;
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::SectionBase<ArraySectionIdTag>>(box)) {
        auto& has_converged =
            get<Convergence::HasConverged>(globalization_result);

        db::mutate<Convergence::Tags::HasConverged<OptionsGroup>,
                   Convergence::Tags::IterationId<OptionsGroup>>(
            make_not_null(&box),
            [&has_converged](
                const gsl::not_null<Convergence::HasConverged*>
                    local_has_converged,
                const gsl::not_null<size_t*> local_iteration_id) noexcept {
              *local_has_converged = std::move(has_converged);
              ++(*local_iteration_id);
            });

        return {std::move(box), false,
                get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                    ? (this_action_index + 2)
                    : (prepare_step_index + 1)};
      }
    }

    if (std::holds_alternative<double>(globalization_result)) {
      if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                   ::Verbosity::Debug)) {
        Parallel::printf(
            "%s " + Options::name<OptionsGroup>() + "(%zu): Globalize(%zu)\n",
            array_index,
            db::get<Convergence::Tags::IterationId<OptionsGroup>>(box),
            db::get<NonlinearSolver::Tags::Globalization<
                Convergence::Tags::IterationId<OptionsGroup>>>(box));
      }

      // Update the step length
      db::mutate<NonlinearSolver::Tags::StepLength<OptionsGroup>,
                 NonlinearSolver::Tags::Globalization<
                     Convergence::Tags::IterationId<OptionsGroup>>>(
          make_not_null(&box),
          [&globalization_result](const gsl::not_null<double*> step_length,
                                  const gsl::not_null<size_t*>
                                      globalization_iteration_id) noexcept {
            *step_length = get<double>(globalization_result);
            ++(*globalization_iteration_id);
          });
      // Continue globalization by taking a step with the updated step length
      // and checking the residual again
      constexpr size_t perform_step_index =
          tmpl::index_of<ActionList, PerformStep<FieldsTag, OptionsGroup, Label,
                                                 ArraySectionIdTag>>::value;
      return {std::move(box), false, perform_step_index};
    }

    // At this point globalization is complete, so we proceed with the algorithm
    auto& has_converged = get<Convergence::HasConverged>(globalization_result);

    db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                             local_has_converged) noexcept {
          *local_has_converged = std::move(has_converged);
        });

    return {std::move(box), false, this_action_index + 1};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct CompleteStep {
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*mete*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + "(%zu): Complete step\n",
          array_index,
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box));
    }

    // Repeat steps until the solve has converged
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList, PrepareStep<FieldsTag, OptionsGroup, Label,
                                               ArraySectionIdTag>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, CompleteStep>::value;
    return {std::move(box), false,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : prepare_step_index};
  }
};

}  // namespace NonlinearSolver::newton_raphson::detail
