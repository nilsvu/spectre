// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::amr::Actions {

struct StopAmr {
 private:
  using iteration_id_tag =
      Convergence::Tags::IterationId<::amr::OptionTags::AmrGroup>;
  using num_iterations_tag =
      Convergence::Tags::Iterations<::amr::OptionTags::AmrGroup>;
  using has_converged_tag =
      Convergence::Tags::HasConverged<::amr::OptionTags::AmrGroup>;

 public:
  using simple_tags = tmpl::list<iteration_id_tag, has_converged_tag>;
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
    // Increment AMR iteration id and determine convergence
    db::mutate<iteration_id_tag, has_converged_tag>(
        [&num_iterations](
            const gsl::not_null<size_t*> iteration_id,
            const gsl::not_null<Convergence::HasConverged*> has_converged) {
          ++(*iteration_id);
          *has_converged =
              Convergence::HasConverged{num_iterations, *iteration_id};
        },
        make_not_null(&box));
    // Do some logging
    if (db::get<iteration_id_tag>(box) > 0 and
        db::get<logging::Tags::Verbosity<::amr::OptionTags::AmrGroup>>(box) >=
            ::Verbosity::Debug) {
      Parallel::printf("%s completed AMR iteration %zu / %zu.\n", element_id,
                       db::get<iteration_id_tag>(box), num_iterations);
    }
    // Stop if num iterations is reached
    return {db::get<has_converged_tag>(box)
                ? Parallel::AlgorithmExecution::Pause
                : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
  }
};

}  // namespace elliptic::amr::Actions
