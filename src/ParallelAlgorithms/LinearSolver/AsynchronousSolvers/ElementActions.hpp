// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver::async_solvers {
template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename Label, typename ArraySectionIdTag, bool ObserveInitial>
struct CompleteStep;
}  // namespace LinearSolver::async_solvers
/// \endcond

/// Functionality shared between parallel linear solvers that have no global
/// synchronization points
namespace LinearSolver::async_solvers {

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>;

struct ResidualReductionFormatter {
  std::string operator()(const size_t iteration_id, const double residual) const
      noexcept {
    if (iteration_id == 0) {
      return "Linear solver '" + solver_name +
             "' initialized with residual: " + get_output(residual);
    } else {
      return "Linear solver '" + solver_name + "' iteration " +
             get_output(iteration_id) +
             " done. Remaining residual: " + get_output(residual);
    }
  }

  void pup(PUP::er& p) noexcept { p | solver_name; }

  std::string solver_name;
};

template <typename OptionsGroup, typename ParallelComponent,
          typename Metavariables, typename ArrayIndex>
void contribute_to_residual_observation(
    const size_t iteration_id, const double residual_magnitude_square,
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const std::string& observation_key_suffix) noexcept {
  auto& local_observer =
      *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
           cache)
           .ckLocalBranch();
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(
          iteration_id,
          pretty_type::get_name<OptionsGroup>() + observation_key_suffix),
      observers::ArrayComponentId{
          std::add_pointer_t<ParallelComponent>{nullptr},
          Parallel::ArrayIndex<ArrayIndex>(array_index)},
      std::string{"/" + Options::name<OptionsGroup>() + observation_key_suffix +
                  "Residuals"},
      std::vector<std::string>{"Iteration", "Residual"},
      reduction_data{iteration_id, residual_magnitude_square},
      UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
               ::Verbosity::Quiet)
          ? std::make_optional(ResidualReductionFormatter{
                Options::name<OptionsGroup>() + observation_key_suffix})
          : std::nullopt);
  if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
               ::Verbosity::Debug)) {
    if (iteration_id == 0) {
      Parallel::printf(
          "Linear solver '" + Options::name<OptionsGroup>() +
              observation_key_suffix +
              "' initialized on element %s. Remaining local residual: %e\n",
          get_output(array_index), sqrt(residual_magnitude_square));
    } else {
      Parallel::printf("Linear solver '" + Options::name<OptionsGroup>() +
                           observation_key_suffix +
                           "' iteration %zu done on element %s. Remaining "
                           "local residual: %e\n",
                       iteration_id, get_output(array_index),
                       sqrt(residual_magnitude_square));
    }
  }
}

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using source_tag = SourceTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<Convergence::Tags::Iterations<OptionsGroup>>;

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
                              operator_applied_to_fields_tag>,
            db::AddComputeTags<
                LinearSolver::Tags::ResidualCompute<fields_tag, source_tag>>>(
            std::move(box),
            // The `PrepareSolve` action populates these tags with initial
            // values, except for `operator_applied_to_fields_tag` which is
            // expected to be updated in every iteration of the algorithm
            std::numeric_limits<size_t>::max(), Convergence::HasConverged{},
            typename operator_applied_to_fields_tag::type{}));
  }
};

template <typename OptionsGroup, typename ArraySectionIdTag = void>
struct RegisterObservers {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) noexcept {
    const auto observation_key_suffix = [&]() noexcept -> std::string {
      if constexpr (std::is_same_v<ArraySectionIdTag, void>) {
        return "";
      } else {
        return db::get<
                   observers::Tags::ObservationKeySuffix<ArraySectionIdTag>>(
                   box)
            .value_or("none");
      }
    }();
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey{pretty_type::get_name<OptionsGroup>() +
                                      observation_key_suffix}};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename ArraySectionIdTag = void>
using RegisterElement = observers::Actions::RegisterWithObservers<
    RegisterObservers<OptionsGroup, ArraySectionIdTag>>;

template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename Label, typename ArraySectionIdTag = void,
          bool ObserveInitial = true>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>;

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
    constexpr size_t iteration_id = 0;

    db::mutate<Convergence::Tags::IterationId<OptionsGroup>,
               Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> local_iteration_id,
           const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t num_iterations) noexcept {
          *local_iteration_id = iteration_id;
          *has_converged =
              Convergence::HasConverged{num_iterations, iteration_id};
        },
        get<Convergence::Tags::Iterations<OptionsGroup>>(box));

    // Observe the initial residual even if no steps are going to be performed
    if constexpr (ObserveInitial) {
      const auto& observation_key_suffix =
          [&]() noexcept -> std::optional<std::string> {
        if constexpr (std::is_same_v<ArraySectionIdTag, void>) {
          return std::make_optional("");
        } else {
          return db::get<
              observers::Tags::ObservationKeySuffix<ArraySectionIdTag>>(box);
        }
      }();
      if (observation_key_suffix) {
        const auto& residual = get<residual_tag>(box);
        const double residual_magnitude_square =
            inner_product(residual, residual);
        contribute_to_residual_observation<OptionsGroup, ParallelComponent>(
            iteration_id, residual_magnitude_square, cache, array_index,
            *observation_key_suffix);
      }
    }

    // Skip steps entirely if the solve has already converged
    constexpr size_t step_end_index =
        tmpl::index_of<ActionList,
                       CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label,
                                    ArraySectionIdTag, ObserveInitial>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, PrepareSolve>::value;
    return {std::move(box), false,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (step_end_index + 1)
                : (this_action_index + 1)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename Label, typename ArraySectionIdTag = void,
          bool ObserveInitial = true>
struct CompleteStep {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Prepare for next iteration
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>,
               Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t num_iterations) noexcept {
          ++(*iteration_id);
          *has_converged =
              Convergence::HasConverged{num_iterations, *iteration_id};
        },
        get<Convergence::Tags::Iterations<OptionsGroup>>(box));

    // Observe element-local residual magnitude
    const auto observation_key_suffix =
        [&]() noexcept -> std::optional<std::string> {
      if constexpr (std::is_same_v<ArraySectionIdTag, void>) {
        return std::make_optional("");
      } else {
        return db::get<
            observers::Tags::ObservationKeySuffix<ArraySectionIdTag>>(box);
      }
    }();
    if (observation_key_suffix) {
      const size_t completed_iterations =
          get<Convergence::Tags::IterationId<OptionsGroup>>(box);
      const auto& residual = get<residual_tag>(box);
      const double residual_magnitude_square =
          inner_product(residual, residual);
      contribute_to_residual_observation<OptionsGroup, ParallelComponent>(
          completed_iterations, residual_magnitude_square, cache, array_index,
          *observation_key_suffix);
    }

    // Repeat steps until the solve has converged
    constexpr size_t step_begin_index =
        tmpl::index_of<ActionList,
                       PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label,
                                    ArraySectionIdTag, ObserveInitial>>::value +
        1;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, CompleteStep>::value;
    return {std::move(box), false,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : step_begin_index};
  }
};

}  // namespace LinearSolver::async_solvers
