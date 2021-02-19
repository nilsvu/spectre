// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"

#include "NumericalAlgorithms/LinearSolver/ExplicitInverse.hpp"

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
struct ResetSubdomainSolver {
  using const_global_cache_tags = tmpl::list<
      LinearSolver::Schwarz::Tags::EnableSubdomainSolverResets<OptionsGroup>>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (get<LinearSolver::Schwarz::Tags::EnableSubdomainSolverResets<
            OptionsGroup>>(box)) {
      if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                   ::Verbosity::Debug)) {
        Parallel::printf("%s " + Options::name<OptionsGroup>() +
                             ": Reset subdomain solver\n",
                         element_id);
      }
      db::mutate<
          LinearSolver::Schwarz::Tags::SubdomainSolverBase<OptionsGroup>>(
          make_not_null(&box), [](const auto subdomain_solver) noexcept {
            // Dereference the gsl::not_null pointer, and then the
            // std::unique_ptr for the subdomain solver's abstract superclass.
            // This needs adjustment if the subdomain solver is stored in the
            // DataBox directly as a derived class and thus there's no
            // std::unique_ptr. Note that std::unique_ptr also has a `reset`
            // function, which must not be confused with the serial linear
            // solver's `reset` function here.
            (*subdomain_solver)->reset();
          });
    }
    return {std::move(box)};
  }
};

}  // namespace LinearSolver::Schwarz::Actions
