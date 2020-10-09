// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"

template <size_t Dim>
struct ElementId;
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}
namespace tuples {
template <typename...>
struct TaggedTuple;
}

namespace LinearSolver::Schwarz::Actions {

template <typename OptionsGroup>
struct ResetSubdomainPreconditioner {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::SubdomainPreconditionerBase<OptionsGroup>>(
        make_not_null(&box), [](const auto stored_preconditioner) noexcept {
          stored_preconditioner->reset();
        });
    return {std::move(box)};
  }
};

}  // namespace LinearSolver::Schwarz::Actions
