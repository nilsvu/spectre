// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
/*!
 * \brief Apply the `boundary_condition` to the `fields_and_fluxes` with
 * arguments from interface tags in the DataBox.
 *
 * This functions assumes the arguments for the `boundary_condition` are stored
 * in the DataBox in tags `domain::Tags::Faces<Dim, Tag>`.
 * This may turn out not to be the most efficient setup, so code that
 * uses the boundary conditions doesn't have to use this function but can
 * procure the arguments differently. For example, future optimizations may
 * involve storing a subset of arguments that don't change during an elliptic
 * solve in direction-maps in the DataBox, and slicing other arguments to the
 * interface every time the boundary conditions are applied.
 *
 * The `ArgsTransform` template parameter can be used to transform the set of
 * argument tags for the boundary conditions further. It must be compatible with
 * `tmpl::transform`. For example, it may wrap the tags in another prefix. Set
 * it to `void` (default) to apply no transformation.
 */
template <bool Linearized, typename ArgsTransform = void, size_t Dim,
          typename Registrars, typename DbTagsList, typename MapKeys,
          typename... FieldsAndFluxes>
void apply_boundary_condition(
    const elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>&
        boundary_condition,
    const db::DataBox<DbTagsList>& box, const MapKeys& map_keys_to_direction,
    const FieldsAndFluxes&... fields_and_fluxes) noexcept {
  call_with_dynamic_type<
      void, typename elliptic::BoundaryConditions::BoundaryCondition<
                Dim, Registrars>::creatable_classes>(
      &boundary_condition,
      [&map_keys_to_direction, &box,
       &fields_and_fluxes...](const auto* const derived) noexcept {
        using Derived = std::decay_t<std::remove_pointer_t<decltype(derived)>>;
        using volume_tags =
            tmpl::conditional_t<Linearized,
                                typename Derived::volume_tags_linearized,
                                typename Derived::volume_tags>;
        using argument_tags = domain::make_faces_tags<
            Dim,
            tmpl::conditional_t<Linearized,
                                typename Derived::argument_tags_linearized,
                                typename Derived::argument_tags>,
            volume_tags>;
        using argument_tags_transformed =
            tmpl::conditional_t<std::is_same_v<ArgsTransform, void>,
                                argument_tags,
                                tmpl::transform<argument_tags, ArgsTransform>>;
        using volume_tags_transformed =
            tmpl::conditional_t<std::is_same_v<ArgsTransform, void>,
                                volume_tags,
                                tmpl::transform<volume_tags, ArgsTransform>>;
        elliptic::util::apply_at<argument_tags_transformed,
                                 volume_tags_transformed>(
            [&derived, &fields_and_fluxes...](const auto&... args) noexcept {
              if constexpr (Linearized) {
                derived->apply_linearized(fields_and_fluxes..., args...);
              } else {
                derived->apply(fields_and_fluxes..., args...);
              }
            },
            box, map_keys_to_direction);
      });
}
}  // namespace elliptic
