// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmNodegroup.hpp"
#include "IO/DataImporter/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"

namespace importer {

namespace detail {
struct InitializeDataFileReader {
  using simple_tags = db::AddSimpleTags<Tags::RegisteredElements>;
  using compute_tags = db::AddComputeTags<>;
  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create<simple_tags>(db::item_type<Tags::RegisteredElements>{}));
  }
};
}  // namespace detail

/*!
 * \brief A nodegroup parallel component that reads in a volume data file and
 * distributes its data to elements of an array parallel component.
 *
 * Each element of the array parallel component must register itself before
 * data can be sent to it. To do so, invoke
 * `importer::Actions::RegisterWithImporter` on each the element. In a
 * subsequent phase you can then invoke
 * `importer::ThreadedActions::ReadElementData` on the `DataFileReader`
 * component to read in the file and distribute its data to the registered
 * elements.
 */
template <class Metavariables>
struct DataFileReader {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename detail::InitializeDataFileReader::return_tag_list>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<detail::InitializeDataFileReader>(
        Parallel::get_parallel_component<DataFileReader>(local_cache));
  }

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};
}  // namespace importer
