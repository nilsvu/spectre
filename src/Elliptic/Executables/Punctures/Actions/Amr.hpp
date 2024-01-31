// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Punctures {
namespace Tags {
using AmrIterationId =
    Convergence::Tags::IterationId<amr::OptionTags::AmrGroup>;
}  // namespace Tags

namespace Actions {

struct ProjectAmrIterationId : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags = tmpl::list<Tags::AmrIterationId>;
  using argument_tags = tmpl::list<>;

  template <typename AmrData>
  static void apply(gsl::not_null<size_t*> /*amr_iteration_id*/,
                    const AmrData& /*unused*/) {}

  template <typename... ParentTags>
  static void apply(gsl::not_null<size_t*> amr_iteration_id,
                    const tuples::TaggedTuple<ParentTags...>& parent_items) {
    *amr_iteration_id = get<Tags::AmrIterationId>(parent_items);
  }
};

struct StopAmr {
 private:
  using num_iterations_tag =
      Convergence::Tags::Iterations<amr::OptionTags::AmrGroup>;

 public:
  using simple_tags = tmpl::list<Tags::AmrIterationId>;
  using const_global_cache_tags = tmpl::list<num_iterations_tag>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t num_iterations = get<num_iterations_tag>(box);
    if (db::get<logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>(box) >=
        ::Verbosity::Quiet) {
      Parallel::printf("%s completed AMR iteration %zu / %zu.\n", element_id,
                       db::get<Tags::AmrIterationId>(box), num_iterations);
    }
    db::mutate<Tags::AmrIterationId>(
        [](const gsl::not_null<size_t*> iteration_id) { ++(*iteration_id); },
        make_not_null(&box));
    return {db::get<Tags::AmrIterationId>(box) > num_iterations
                ? Parallel::AlgorithmExecution::Pause
                : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
  }
};

struct GoToStart {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*element_id*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Continue, 0};
  }
};

}  // namespace Actions
}  // namespace Punctures
