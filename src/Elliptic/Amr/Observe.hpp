// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "Elliptic/Amr/Tags.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Functional.hpp"

namespace elliptic::amr::observe_detail {

using reduction_data = Parallel::ReductionData<
    // AMR level
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Error
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;

struct Registration {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey{Options::name<OptionTags::AmrGroup>()}};
  }
};

template <typename ParallelComponent, typename Metavariables>
void contribute_to_reduction_observer(
    const size_t level, const double error,
    Parallel::GlobalCache<Metavariables>& cache) noexcept {
  const auto observation_id =
      observers::ObservationId(level, Options::name<OptionTags::AmrGroup>());
  auto& reduction_writer = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache);
  auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
  Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
      // Node 0 is always the writer, so directly call the component on that
      // node
      reduction_writer[0], observation_id,
      static_cast<size_t>(Parallel::my_node(*my_proxy.ckLocal())),
      std::string{"/" + Options::name<OptionTags::AmrGroup>()},
      std::vector<std::string>{"Level", "Error"}, reduction_data{level, error});
}

}  // namespace elliptic::amr::observe_detail
