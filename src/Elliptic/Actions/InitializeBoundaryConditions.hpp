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
#include "Elliptic/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic {
namespace Actions {

struct InitializeBoundaryConditions {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    const auto& boundary_conditions_provider =
        db::get<typename Metavariables::boundary_conditions_tag>(box);

    std::unordered_map<
        Direction<Dim>,
        tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
            elliptic::Tags::BoundaryCondition, typename system::primal_fields>>>
        boundary_conditions{};
    std::unordered_map<Direction<Dim>,
                       tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
                           elliptic::Tags::BoundaryCondition,
                           typename Metavariables::primal_variables>>>
        lin_boundary_conditions{};
    for (const auto& direction :
         db::get<domain::Tags::Element<Dim>>(box).external_boundaries()) {
      tmpl::for_each<typename system::primal_fields>([&](auto tag_v) noexcept {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<elliptic::Tags::BoundaryCondition<tag>>(
            boundary_conditions[direction]) =
            boundary_conditions_provider.boundary_condition_type(
                db::get<domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsExterior<Dim>,
                    domain::Tags::Coordinates<Dim, Frame::Inertial>>>(box)
                    .at(direction),
                direction, tag{});
        using lin_tag =
            tmpl::at<typename Metavariables::primal_variables,
                     tmpl::index_of<typename system::primal_fields, tag>>;
        get<elliptic::Tags::BoundaryCondition<lin_tag>>(
            lin_boundary_conditions[direction]) =
            get<elliptic::Tags::BoundaryCondition<tag>>(
                boundary_conditions[direction]);
      });
    }

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeBoundaryConditions,
            db::AddSimpleTags<
                domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsExterior<Dim>,
                    elliptic::Tags::BoundaryConditions<
                        typename system::primal_fields>>,
                domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsExterior<Dim>,
                    elliptic::Tags::BoundaryConditions<
                        typename Metavariables::primal_variables>>>>(
            std::move(box), std::move(boundary_conditions),
            std::move(lin_boundary_conditions)));
  }
};

}  // namespace Actions
}  // namespace elliptic
