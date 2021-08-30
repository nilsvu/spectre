// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Amr/ErrorMonitorActions.hpp"
#include "Elliptic/Amr/Tags.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GetSection.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace elliptic::amr::detail {
template <typename Metavariables, size_t Dim>
struct ErrorMonitor;
template <size_t Dim, typename ArraySectionIdTag>
struct Complete;
}  // namespace elliptic::amr::detail
/// \endcond

namespace elliptic::amr::detail {

template <size_t Dim, typename ArraySectionIdTag>
struct Prepare {
  using simple_tags =
      tmpl::list<Tags::Level, Tags::HasConverged, Tags::ParentMesh<Dim>>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionTags::AmrGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionTags::AmrGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s Prepare AMR.\n", array_index);
    }
    db::mutate<Tags::Level, Tags::HasConverged>(
        make_not_null(&box), [](const gsl::not_null<size_t*> level,
                                const gsl::not_null<Convergence::HasConverged*>
                                    has_converged) noexcept {
          *level = 1;
          *has_converged = Convergence::HasConverged{};
        });
    return {std::move(box)};
  }
};

template <size_t Dim, typename ArraySectionIdTag>
struct ContributeError {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Contribute error only from elements that are part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        return {std::move(box)};
      }
    }
    const size_t level = db::get<Tags::Level>(box);
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionTags::AmrGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s AMR(%zu) contribute error\n", array_index, level);
    }
    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<MeasureError<Dim, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>>{level},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ErrorMonitor<Metavariables, Dim>>(
            cache),
        make_not_null(&section));
    return {std::move(box)};
  }
};

template <size_t Dim, typename ArraySectionIdTag>
struct RefineMesh {
  using inbox_tags = tmpl::list<InboxTags::ErrorMeasurement>;
  using const_global_cache_tags = tmpl::list<Tags::IncreaseNumPointsUniformly>;

  template <typename DbTagsList, typename... AllInboxTags,
            typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<AllInboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& array_index, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    // Scoped because `Tags::Level` is mutated below
    {
      const size_t level = db::get<Tags::Level>(box);
      auto& inbox = get<InboxTags::ErrorMeasurement>(inboxes);
      if (inbox.find(level) == inbox.end()) {
        return {std::move(box), Parallel::AlgorithmExecution::Retry,
                std::numeric_limits<size_t>::max()};
      }
      auto received_data = std::move(inbox.extract(level).mapped());
      // const double error = get<0>(received_data);
      auto& has_converged = get<1>(received_data);
      db::mutate<Tags::HasConverged>(
          make_not_null(&box),
          [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                               local_has_converged) noexcept {
            *local_has_converged = std::move(has_converged);
          });

      // Skip to the end if AMR has converged
      if (get<Tags::HasConverged>(box)) {
        constexpr size_t completion_index =
            tmpl::index_of<ActionList, Complete<Dim, ArraySectionIdTag>>::value;
        return {std::move(box), Parallel::AlgorithmExecution::Continue,
                completion_index + 1};
      }
    }

    // The AMR level is complete. Now, refine the mesh for the next level.
    db::mutate<Tags::Level>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> local_level) noexcept {
          ++(*local_level);
        });
    const size_t level = db::get<Tags::Level>(box);
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionTags::AmrGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s AMR(%zu): Refine mesh.\n", array_index, level);
    }
    // - Currently limited to p-refinement
    db::mutate<Tags::ParentMesh<Dim>>(
        make_not_null(&box),
        [](const gsl::not_null<std::optional<Mesh<Dim>>*> parent_mesh,
           const Mesh<Dim>& mesh) noexcept { *parent_mesh = mesh; },
        db::get<domain::Tags::Mesh<Dim>>(box));
    db::mutate<domain::Tags::InitialExtents<Dim>>(
        make_not_null(&box),
        [](const gsl::not_null<std::vector<std::array<size_t, Dim>>*>
               initial_extents,
           const size_t add_p) noexcept {
          for (auto& extents : *initial_extents) {
            for (size_t d = 0; d < Dim; ++d) {
              gsl::at(extents, d) += add_p;
            }
          }
        },
        db::get<Tags::IncreaseNumPointsUniformly>(box));
    // Continue with re-initialization of the DataBox
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, RefineMesh<Dim, ArraySectionIdTag>>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            this_action_index + 1};
  }
};

template <size_t Dim, typename ArraySectionIdTag>
struct Complete {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    // Once AMR is done, proceed with the action list
    if (db::get<Tags::HasConverged>(box)) {
      constexpr size_t this_action_index =
          tmpl::index_of<ActionList, Complete<Dim, ArraySectionIdTag>>::value;
      return {std::move(box), Parallel::AlgorithmExecution::Continue,
              this_action_index + 1};
    }
    // AMR is not done yet. Loop around to the beginning of the `SolveActions`.
    constexpr size_t solve_index =
        tmpl::index_of<ActionList, Prepare<Dim, ArraySectionIdTag>>::value + 1;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            solve_index};
  }
};

}  // namespace elliptic::amr::detail
