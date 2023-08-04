// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// In this test we create a few array elements, partition them in sections based
// on their index, and then count the elements in each section separately in a
// reduction.

enum class EvenOrOdd { Even, Odd };

// These don't need to be DataBox tags because they aren't placed in the
// DataBox. they are used to identify the section. Note that in many practical
// applications the section ID tag _is_ placed in the DataBox nonetheless.
struct EvenOrOddTag {
  using type = EvenOrOdd;
};
struct IsFirstElementTag {
  using type = bool;
};

template <typename ArraySectionIdTag>
struct ReceiveCount;

template <typename ArraySectionIdTag>
struct Count {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      const ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    const bool section_id = [&array_index]() {
      if constexpr (std::is_same_v<ArraySectionIdTag, EvenOrOddTag>) {
        return array_index % 2 == 0;
      } else {
        return array_index == 0;
      }
    }();
    // [section_reduction]
    auto& array_section = db::get_mutable_reference<
        Parallel::Tags::Section<ParallelComponent, ArraySectionIdTag>>(
        make_not_null(&box));
    if (array_section.has_value()) {
      // We'll just count the elements in each section
      Parallel::ReductionData<
          Parallel::ReductionDatum<bool, funcl::AssertEqual<>>,
          Parallel::ReductionDatum<size_t, funcl::Plus<>>>
          reduction_data{section_id, 1};
      // Reduce over the section and broadcast to the full array
      auto& array_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::contribute_to_reduction<ReceiveCount<ArraySectionIdTag>>(
          std::move(reduction_data), array_proxy[array_index], array_proxy,
          make_not_null(&*array_section));
    }
    // [section_reduction]
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename ArraySectionIdTag>
struct ReceiveCount {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const int array_index, const bool section_id,
                    const size_t count) {
    if constexpr (std::is_same_v<ArraySectionIdTag, EvenOrOddTag>) {
      const bool is_even = section_id;
      Parallel::printf(
          "Element %d received reduction: Counted %zu %s elements.\n",
          array_index, count, is_even ? "even" : "odd");
      SPECTRE_PARALLEL_REQUIRE(count == (is_even ? 3 : 2));
    } else {
      const bool is_first_element = section_id;
      Parallel::printf(
          "Element %d received reduction: Counted %zu element in "
          "'IsFirstElement' section.\n",
          array_index, count);
      SPECTRE_PARALLEL_REQUIRE(is_first_element);
      SPECTRE_PARALLEL_REQUIRE(count == 1);
    }
  }
};

template <typename Metavariables>
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Count<EvenOrOddTag>,
                     // Test performing a reduction over another section
                     Count<IsFirstElementTag>,
                     // Test performing multiple reductions asynchronously
                     Count<EvenOrOddTag>, Count<EvenOrOddTag>,
                     Count<IsFirstElementTag>,
                     Parallel::Actions::TerminatePhase>>>;

  // [sections_example]
  using array_allocation_tags = tmpl::list<
      // The section proxy will be stored in each element's DataBox in this tag
      // for convenient access
      Parallel::Tags::Section<ArrayComponent, EvenOrOddTag>,
      Parallel::Tags::Section<ArrayComponent, IsFirstElementTag>>;
  using simple_tags_from_options =
      tmpl::append<Parallel::get_simple_tags_from_options<
                       Parallel::get_initialization_actions_list<
                           phase_dependent_action_list>>,
                   array_allocation_tags>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      tuples::tagged_tuple_from_typelist<simple_tags_from_options>
          initialization_items,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayComponent>(local_cache);

    // Create sections from element IDs (the corresponding elements will be
    // created below)
    const size_t num_elements = 5;
    std::vector<CkArrayIndex> even_elements{};
    std::vector<CkArrayIndex> odd_elements{};
    for (size_t i = 0; i < num_elements; ++i) {
      (i % 2 == 0 ? even_elements : odd_elements)
          .push_back(Parallel::ArrayIndex<int>(static_cast<int>(i)));
    }
    std::vector<CkArrayIndex> first_element{};
    first_element.push_back(Parallel::ArrayIndex<int>(0));
    using EvenOrOddSection = Parallel::Section<ArrayComponent, EvenOrOddTag>;
    const EvenOrOddSection even_section{
        EvenOrOdd::Even, EvenOrOddSection::cproxy_section::ckNew(
                             array_proxy.ckGetArrayID(), even_elements.data(),
                             even_elements.size())};
    const EvenOrOddSection odd_section{
        EvenOrOdd::Odd, EvenOrOddSection::cproxy_section::ckNew(
                            array_proxy.ckGetArrayID(), odd_elements.data(),
                            odd_elements.size())};
    using IsFirstElementSection =
        Parallel::Section<ArrayComponent, IsFirstElementTag>;
    const IsFirstElementSection is_first_element_section{
        true, IsFirstElementSection::cproxy_section::ckNew(
                  array_proxy.ckGetArrayID(), first_element.data(),
                  first_element.size())};

    // Create array elements, copying the appropriate section proxy into their
    // DataBox
    const size_t number_of_procs = static_cast<size_t>(sys::number_of_procs());
    size_t which_proc = 0;
    for (size_t i = 0; i < 5; ++i) {
      tuples::get<Parallel::Tags::Section<ArrayComponent, EvenOrOddTag>>(
          initialization_items) = i % 2 == 0 ? even_section : odd_section;
      tuples::get<Parallel::Tags::Section<ArrayComponent, IsFirstElementTag>>(
          initialization_items) =
          i == 0 ? std::make_optional(is_first_element_section) : std::nullopt;
      while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
      array_proxy[static_cast<int>(i)].insert(global_cache,
                                              initialization_items, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();
  }
  // [sections_example]

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ArrayComponent>(local_cache)
        .start_phase(next_phase);
  }
};

struct Metavariables {
  using component_list = tmpl::list<ArrayComponent<Metavariables>>;

  static constexpr Options::String help = "Test section reductions";

  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Testing,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
