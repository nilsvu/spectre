// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic {
namespace Actions {

template <typename System, typename BoundaryConditionsTag,
          typename PrimalVariables>
struct InitializeBoundaryConditionTypes {
 private:
  static constexpr size_t Dim = System::volume_dim;
  using primal_fields = typename System::primal_fields;
  using primal_variables = PrimalVariables;

 public:
  using simple_tags = tmpl::flatten<tmpl::list<
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsExterior<Dim>,
          elliptic::Tags::BoundaryConditionTypes<primal_fields>>,
      tmpl::conditional_t<
          std::is_same_v<primal_fields, primal_variables>, tmpl::list<>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsExterior<Dim>,
              elliptic::Tags::BoundaryConditionTypes<primal_variables>>>>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& boundary_conditions = db::get<BoundaryConditionsTag>(box);

    std::unordered_map<
        Direction<Dim>,
        tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
            elliptic::Tags::BoundaryConditionType, primal_fields>>>
        boundary_condition_types{};
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& direction : element.external_boundaries()) {
      const auto& inertial_coords = db::get<domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsExterior<Dim>,
          domain::Tags::Coordinates<Dim, Frame::Inertial>>>(box)
                                        .at(direction);
      tmpl::for_each<primal_fields>([&direction, &boundary_condition_types,
                                     &boundary_conditions,
                                     &inertial_coords](auto tag_v) noexcept {
        using field_tag = tmpl::type_from<decltype(tag_v)>;
        get<elliptic::Tags::BoundaryConditionType<tag>>(
            boundary_condition_types[direction]) =
            boundary_conditions.boundary_condition_type(inertial_coords,
                                                        direction, field_tag{});
      });
    }
    ::Initialization::mutate_assign<domain::Tags::Interface<
        domain::Tags::BoundaryDirectionsExterior<Dim>,
        elliptic::Tags::BoundaryConditionTypes<primal_fields>>>(
        make_not_null(&box), std::move(boundary_conditions));

    if constexpr (not std::is_same_v<primal_fields, primal_variables>) {
      std::unordered_map<
          Direction<Dim>,
          tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
              elliptic::Tags::BoundaryConditionType, primal_variables>>>
          vars_boundary_condition_types{};
      for (const auto& [direction, bc_types_in_direction] :
           boundary_condition_types) {
        tmpl::for_each<primal_fields>([&](auto tag_v) noexcept {
          using field_tag = tmpl::type_from<decltype(tag_v)>;
          using var_tag = tmpl::at<primal_variables,
                                   tmpl::index_of<primal_fields, field_tag>>;
          get<elliptic::Tags::BoundaryConditionType<var_tag>>(
              vars_boundary_condition_types[direction]) =
              get<elliptic::Tags::BoundaryConditionType<field_tag>>(
                  bc_types_in_direction);
        });
      }
      ::Initialization::mutate_assign<domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsExterior<Dim>,
          elliptic::Tags::BoundaryConditionTypes<primal_variables>>>(
          make_not_null(&box), std::move(vars_boundary_condition_types));
    }

    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace elliptic
