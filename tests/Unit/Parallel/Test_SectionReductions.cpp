// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/Options.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Tags/Section.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// In this test we create a few array elements, partition them in sections based
// on their index being even or odd, and then count the elements in each section
// separately in a reduction.

enum class EvenOrOdd { Even, Odd };

// This doesn't need to be a DataBox tag because it isn't placed in the DataBox.
// It's used to identify the section. Note that in many practical applications
// the section ID tag _is_ placed in the DataBox nonetheless.
struct EvenOrOddTag {
  using type = EvenOrOdd;
};

template <typename ArraySectionIdTag>
struct ReceiveCount;

template <typename ArraySectionIdTag>
struct Count {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // [section_reduction]
    // We'll just count the elements in each section. The sections hold "even"
    // and "odd" elements.
    Parallel::ReductionData<
        Parallel::ReductionDatum<bool, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<size_t, funcl::Plus<>>>
        reduction_data{array_index % 2 == 0, 1};
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& array_section =
        *db::get<Parallel::Tags::Section<ParallelComponent, ArraySectionIdTag>>(
            box);
    // Reduce over the section and broadcast to the full array
    Parallel::contribute_to_reduction<ReceiveCount<ArraySectionIdTag>>(
        std::move(reduction_data), array_proxy[array_index], array_proxy,
        array_section);
    // [section_reduction]
    return {std::move(box), true};
  }
};

template <typename ArraySectionIdTag>
struct ReceiveCount {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const int array_index, const int is_even,
                    const size_t count) noexcept {
    Parallel::printf(
        "Element %d received reduction: Counted %zu %s elements.\n",
        array_index, count, is_even ? "even" : "odd");
    SPECTRE_PARALLEL_REQUIRE(count == (is_even ? 3 : 2));
  }
};

template <typename Metavariables>
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>,
                 Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::TestReductions,
                                        tmpl::list<Count<EvenOrOddTag>>>>;

  // [sections_example]
  // List all section ID tags
  using section_id_tags = tmpl::list<EvenOrOddTag>;
  using array_allocation_tags = tmpl::list<
      // The section proxy will be stored in each element's DataBox in this tag
      // for convenient access
      Parallel::Tags::Section<ArrayComponent, EvenOrOddTag>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      tuples::tagged_tuple_from_typelist<initialization_tags>
          initialization_items) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
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
    using EvenOrOddSection = Parallel::Section<ArrayComponent, EvenOrOddTag>;
    const EvenOrOddSection even_section{
        EvenOrOdd::Even, EvenOrOddSection::cproxy_section::ckNew(
                             array_proxy.ckGetArrayID(), even_elements.data(),
                             even_elements.size())};
    const EvenOrOddSection odd_section{
        EvenOrOdd::Odd, EvenOrOddSection::cproxy_section::ckNew(
                            array_proxy.ckGetArrayID(), odd_elements.data(),
                            odd_elements.size())};

    // Create array elements, copying the appropriate section proxy into their
    // DataBox
    const int number_of_procs = sys::number_of_procs();
    for (size_t i = 0; i < 5; ++i) {
      tuples::get<Parallel::Tags::Section<ArrayComponent, EvenOrOddTag>>(
          initialization_items) = i % 2 == 0 ? even_section : odd_section;
      array_proxy[static_cast<int>(i)].insert(
          global_cache, initialization_items,
          static_cast<int>(i) % number_of_procs);
    }
    array_proxy.doneInserting();
  }
  // [sections_example]

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ArrayComponent>(local_cache)
        .start_phase(next_phase);
  }
};

struct Metavariables {
  using component_list = tmpl::list<ArrayComponent<Metavariables>>;

  static constexpr Options::String help = "Test section reductions";

  enum class Phase { Initialization, TestReductions, Exit };

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
      /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::TestReductions;
      case Phase::TestReductions:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR("Unknown Phase...");
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
