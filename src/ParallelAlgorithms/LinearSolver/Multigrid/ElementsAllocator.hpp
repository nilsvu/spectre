// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Serialize.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Hierarchy.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Printf.hpp"

namespace LinearSolver::multigrid {

template <size_t Dim, typename OptionsGroup>
struct ElementsAllocator {
  using section_id_tags = tmpl::list<Tags::MultigridLevel, Tags::IsFinestGrid>;

  template <typename ElementArray>
  using array_allocation_tags =
      tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                 Tags::ChildrenRefinementLevels<Dim>,
                 Tags::ParentRefinementLevels<Dim>,
                 Parallel::Tags::Section<ElementArray, Tags::MultigridLevel>,
                 Parallel::Tags::Section<ElementArray, Tags::IsFinestGrid>,
                 Tags::MaxLevels<OptionsGroup>>;

  template <typename ElementArray, typename Metavariables,
            typename... InitializationTags>
  static void apply(Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
                    const tuples::TaggedTuple<InitializationTags...>&
                        initialization_items_) noexcept {
    auto initialization_items =
        deserialize<tuples::TaggedTuple<InitializationTags...>>(
            serialize<tuples::TaggedTuple<InitializationTags...>>(
                initialization_items_)
                .data());
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);
    const auto& domain = Parallel::get<domain::Tags::Domain<Dim>>(local_cache);
    auto& initial_refinement_levels =
        get<domain::Tags::InitialRefinementLevels<Dim>>(initialization_items);
    auto& children_refinement_levels =
        get<Tags::ChildrenRefinementLevels<Dim>>(initialization_items);
    auto& parent_refinement_levels =
        get<Tags::ParentRefinementLevels<Dim>>(initialization_items);
    const auto& max_levels =
        get<Tags::MaxLevels<OptionsGroup>>(initialization_items);
    size_t multigrid_level = 0;
    do {
      // Store the grid as base before coarsening it
      children_refinement_levels = initial_refinement_levels;
      initial_refinement_levels = parent_refinement_levels;
      // Construct coarsened (parent) grid
      if (not max_levels.has_value() or multigrid_level < *max_levels - 1) {
        parent_refinement_levels =
            LinearSolver::multigrid::coarsen(initial_refinement_levels);
      }
      // Create element IDs for all elements on this level
      std::vector<ElementId<Dim>> all_element_ids{};
      for (const auto& block : domain.blocks()) {
        const std::vector<ElementId<Dim>> element_ids = initial_element_ids(
            block.id(), initial_refinement_levels[block.id()], multigrid_level);
        all_element_ids.insert(all_element_ids.begin(), element_ids.begin(),
                               element_ids.end());
      }
      // Create an array section for this refinement level
      std::vector<CkArrayIndex> all_array_indices(all_element_ids.size());
      std::transform(all_element_ids.begin(), all_element_ids.end(),
                     all_array_indices.begin(),
                     [](const ElementId<Dim>& element_id) noexcept {
                       return Parallel::ArrayIndex<ElementId<Dim>>(element_id);
                     });
      using MultigridLevelSection =
          Parallel::Section<ElementArray, Tags::MultigridLevel>;
      get<Parallel::Tags::Section<ElementArray, Tags::MultigridLevel>>(
          initialization_items) = MultigridLevelSection{
          multigrid_level,
          MultigridLevelSection::cproxy_section::ckNew(
              element_array.ckGetArrayID(), all_array_indices.data(),
              all_array_indices.size())};
      using FinestGridSection =
          Parallel::Section<ElementArray, Tags::IsFinestGrid>;
      get<Parallel::Tags::Section<ElementArray, Tags::IsFinestGrid>>(
          initialization_items) =
          multigrid_level == 0
              ? std::make_optional(FinestGridSection{
                    true,
                    FinestGridSection::cproxy_section::ckNew(
                        element_array.ckGetArrayID(), all_array_indices.data(),
                        all_array_indices.size())})
              : std::nullopt;
      // Create the elements for this refinement level and distribute them among
      // processors
      const int number_of_procs = sys::number_of_procs();
      const domain::BlockZCurveProcDistribution<Dim> element_distribution{
          static_cast<size_t>(number_of_procs), initial_refinement_levels};
      for (const auto& element_id : all_element_ids) {
        const size_t target_proc = element_distribution.get_proc_for_element(
            element_id.block_id(), element_id);
        element_array(element_id)
            .insert(global_cache, initialization_items, target_proc);
      }
      Parallel::printf(
          "%s level %zu has %zu elements in %zu blocks distributed on %d "
          "procs.\n",
          Options::name<OptionsGroup>(), multigrid_level,
          all_element_ids.size(), domain.blocks().size(), number_of_procs);
      ++multigrid_level;
    } while (initial_refinement_levels != parent_refinement_levels);
    element_array.doneInserting();
  }
};  // namespace LinearSolver::multigrid

}  // namespace LinearSolver::multigrid
