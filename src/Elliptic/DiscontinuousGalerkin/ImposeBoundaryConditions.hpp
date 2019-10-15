// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Packages data on external boundaries so that they represent
 * homogeneous (zero) Dirichlet boundary conditions.
 *
 * This action imposes homogeneous boundary conditions on all fields in
 * `system::primal_fields`. The fields are wrapped in
 * `LinearSolver::Tags::Operand`. The result should be a subset of the
 * `system::variables`. Because we are working with the linear solver operand,
 * we cannot impose non-zero boundary conditions here. Instead, non-zero
 * boundary conditions are handled as contributions to the linear solver source
 * during initialization.
 *
 * \warning This actions works only for scalar fields right now. It should be
 * considered a temporary solution and will have to be reworked for more
 * involved boundary conditions.
 *
 * With:
 * - `interior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 * - `exterior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>, Tag>`
 *
 * Uses:
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *   - `primal_fields`
 * - DataBox:
 *   - `interior<variables_tag>`
 *
 * DataBox changes:
 * - Modifies:
 *   - `exterior<variables_tag>`
 */
template <typename Metavariables>
struct ImposeHomogeneousDirichletBoundaryConditions {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using dirichlet_tags = typename system::primal_fields;
    constexpr size_t volume_dim = system::volume_dim;

    // Set the data on exterior (ghost) faces to impose the boundary conditions
    db::mutate<::Tags::Interface<::Tags::BoundaryDirectionsExterior<volume_dim>,
                                 typename system::variables_tag>>(
        make_not_null(&box),
        // Need to use system::volume_dim below instead of just
        // volume_dim to avoid an ICE on gcc 7.
        [](const gsl::not_null<db::item_type<::Tags::Interface<
               ::Tags::BoundaryDirectionsExterior<system::volume_dim>,
               typename system::variables_tag>>*>
               exterior_boundary_vars,
           const db::const_item_type<::Tags::Interface<
               ::Tags::BoundaryDirectionsInterior<system::volume_dim>,
               typename system::variables_tag>>& interior_vars) noexcept {
          for (auto& exterior_direction_and_vars : *exterior_boundary_vars) {
            auto& direction = exterior_direction_and_vars.first;
            auto& exterior_vars = exterior_direction_and_vars.second;

            // By default, use the variables on the external boundary for the
            // exterior
            exterior_vars = interior_vars.at(direction);

            // For those variables where we have boundary conditions, impose
            // zero Dirichlet b.c. here. The non-zero boundary conditions are
            // handled as contributions to the source in InitializeElement.
            // Imposing them here would not work because we are working with the
            // linear solver operand.
            tmpl::for_each<dirichlet_tags>([&exterior_vars](
                auto dirichlet_tag_val) noexcept {
              using dirichlet_tag =
                  tmpl::type_from<decltype(dirichlet_tag_val)>;
              using dirichlet_operand_tag =
                  LinearSolver::Tags::Operand<dirichlet_tag>;
              // Use mirror principle
              auto& exterior_dirichlet_field =
                  get<dirichlet_operand_tag>(exterior_vars);
              for (size_t i = 0; i < exterior_dirichlet_field.size(); i++) {
                exterior_dirichlet_field[i] *= -1.;
              }
            });
          }
        },
        get<::Tags::Interface<::Tags::BoundaryDirectionsInterior<volume_dim>,
                              typename system::variables_tag>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
