// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace LinearSolver {

namespace none_detail {
struct DoNothing {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("Doing nothing\n");
    return {std::move(box), false};
  }
};
}  // namespace none_detail

template <typename FieldsTag, typename OptionsGroup,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct None {
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using options_group = OptionsGroup;
  using component_list = tmpl::list<>;
  using observed_reduction_data_tags = tmpl::list<>;

  struct initialize_element {
    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ArrayIndex, typename ActionList,
              typename ParallelComponent>
    static auto apply(db::DataBox<DbTagsList>& box,
                      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                      const ArrayIndex& /*array_index*/,
                      const ActionList /*meta*/,
                      const ParallelComponent* const /*meta*/) noexcept {
      using compute_tags = db::AddComputeTags<>;
      return std::make_tuple(
          ::Initialization::merge_into_databox<
              initialize_element,
              db::AddSimpleTags<LinearSolver::Tags::HasConverged<OptionsGroup>>,
              compute_tags>(std::move(box), Convergence::HasConverged{}));
    }
  };

  struct prepare_solve {
    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ArrayIndex, typename ActionList,
              typename ParallelComponent>
    static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
        db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
      Parallel::printf("Prepare identity\n");
      db::mutate<LinearSolver::Tags::HasConverged<OptionsGroup>>(
          make_not_null(&box),
          [](const gsl::not_null<Convergence::HasConverged*>
                 has_converged) noexcept {
            *has_converged = Convergence::HasConverged{};
          });
      return {std::move(box), false};
    }
  };

  using prepare_step = none_detail::DoNothing;

  struct perform_step {
    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ArrayIndex, typename ActionList,
              typename ParallelComponent>
    static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
        db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
      Parallel::printf("Identity\n");
      db::mutate<fields_tag, LinearSolver::Tags::HasConverged<OptionsGroup>>(
          make_not_null(&box),
          [](const gsl::not_null<db::item_type<fields_tag>*> fields,
             const gsl::not_null<Convergence::HasConverged*> has_converged,
             const db::item_type<source_tag>& source) noexcept {
            *fields = db::item_type<fields_tag>(source);
            *has_converged = Convergence::HasConverged{{1, 0., 0.}, 1, 1., 1.};
          },
          get<source_tag>(box));
      return {std::move(box), false};
    }
  };
};
}  // namespace LinearSolver
