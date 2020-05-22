// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic {
namespace dg {

/*!
 * \brief Set the `exterior_vars` so that they represent homogeneous (zero)
 * boundary conditions.
 *
 * To impose homogeneous boundary conditions we mirror the `interior_vars` and
 * invert their sign. Variables that are not in the `BoundaryConditionTags` list
 * are mirrored without changing their sign, so no boundary conditions are
 * imposed on them.
 */
template <typename BoundaryConditionTags, typename TagsList>
void homogeneous_boundary_conditions(
    const gsl::not_null<Variables<TagsList>*> exterior_vars,
    const Variables<TagsList>& interior_vars) noexcept {
  // By default, use the variables on the external boundary for the
  // exterior
  *exterior_vars = interior_vars;
  // For those variables where we have boundary conditions, impose zero b.c.
  // here. The non-zero boundary conditions are handled as contributions to the
  // source in InitializeElement. Imposing them here would break linearity of
  // the DG operator.
  tmpl::for_each<BoundaryConditionTags>([&exterior_vars](auto tag_v) noexcept {
    using tag = tmpl::type_from<decltype(tag_v)>;
    // Use mirror principle
    auto& exterior_field = get<tag>(*exterior_vars);
    for (size_t i = 0; i < exterior_field.size(); i++) {
      exterior_field[i] *= -1.;
    }
  });
}

namespace Actions {

/*!
 * \brief Set field data on external boundaries so that they represent
 * homogeneous (zero) boundary conditions.
 *
 * This action imposes homogeneous boundary conditions on all
 * `BoundaryConditionTags`
 *
 * \see `elliptic::dg::homogeneous_boundary_conditions`
 *
 * \note We cannot impose inhomogeneous boundary conditions here because it
 * would break linearity of the DG operator: If in the system of equations
 * \f$A(x)=b\f$ the DG operator \f$A\f$ had non-zero boundary contributions then
 * \f$A(x=0)\neq 0\f$, which breaks linearity. Instead, inhomogeneous
 * boundary conditions are handled as contributions to the source \f$b\f$
 * during initialization.
 *
 * With:
 * - `interior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 * - `exterior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>, Tag>`
 *
 * Uses:
 * - DataBox:
 *   - `interior<VariablesTag>`
 *
 * DataBox changes:
 * - Modifies:
 *   - `exterior<VariablesTag>`
 */
template <typename VariablesTag, typename PrimalFields,
          typename AuxiliaryFields>
struct ImposeHomogeneousBoundaryConditions {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Set the data on exterior (ghost) faces to impose the boundary conditions
    db::mutate<domain::Tags::Interface<
        domain::Tags::BoundaryDirectionsExterior<Dim>, VariablesTag>>(
        make_not_null(&box),
        [](const auto exterior_boundary_vars, const auto& interior_vars,
           const std::unordered_map<Direction<Dim>,
                                    elliptic::BoundaryCondition>&
               boundary_condition_types) noexcept {
          for (auto& exterior_direction_and_vars : *exterior_boundary_vars) {
            auto& direction = exterior_direction_and_vars.first;
            switch (boundary_condition_types.at(direction)) {
              case elliptic::BoundaryCondition::Dirichlet:
                homogeneous_boundary_conditions<PrimalFields>(
                    make_not_null(&exterior_direction_and_vars.second),
                    interior_vars.at(direction));
                break;
              case elliptic::BoundaryCondition::Neumann:
                homogeneous_boundary_conditions<AuxiliaryFields>(
                    make_not_null(&exterior_direction_and_vars.second),
                    interior_vars.at(direction));
                break;
              default:
                ERROR("Invalid case");
            }
          }
        },
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>, VariablesTag>>(box),
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsExterior<Dim>,
            elliptic::Tags::BoundaryCondition>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
