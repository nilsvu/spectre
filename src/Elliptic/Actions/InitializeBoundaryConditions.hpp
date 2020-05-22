// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic {
namespace Actions {

template <typename BoundaryConditionsProviderTag>
struct InitializeBoundaryConditions {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& boundary_conditions_provider =
        db::get<BoundaryConditionsProviderTag>(box);

    std::unordered_map<Direction<Dim>, elliptic::BoundaryCondition>
        boundary_conditions{};
    for (const auto& direction :
         db::get<domain::Tags::Element<Dim>>(box).external_boundaries()) {
      boundary_conditions[direction] =
          boundary_conditions_provider.boundary_condition_type(
              db::get<domain::Tags::Interface<
                  domain::Tags::BoundaryDirectionsExterior<Dim>,
                  domain::Tags::Coordinates<Dim, Frame::Inertial>>>(box)
                  .at(direction),
              direction);
    }

    return std::make_tuple(::Initialization::merge_into_databox<
                           InitializeBoundaryConditions,
                           db::AddSimpleTags<domain::Tags::Interface<
                               domain::Tags::BoundaryDirectionsExterior<Dim>,
                               elliptic::Tags::BoundaryCondition>>>(
        std::move(box), std::move(boundary_conditions)));
  }
};

}  // namespace Actions
}  // namespace elliptic
