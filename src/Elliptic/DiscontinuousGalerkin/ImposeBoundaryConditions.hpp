// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::dg::Actions {

template <typename BoundaryConditionsTag, typename FieldsTag,
          typename PrimalFields, typename AuxiliaryFields,
          typename FluxesComputerTag>
struct ImposeBoundaryConditions {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& boundary_conditions = get<BoundaryConditionsTag>(box);
    using BoundaryConditions = std::decay_t<decltype(boundary_conditions)>;
    const auto& fluxes_computer = get<FluxesComputerTag>(box);
    using FluxesComputer = std::decay_t<decltype(fluxes_computer)>;
    const auto& element = get<domain::Tags::Element<Dim>>(box);

    for (const auto& direction : element.external_boundaries()) {
      // Get fluxes args out of DataBox
      const auto fluxes_args = db::apply_at<
          tmpl::transform<
              typename FluxesComputer::argument_tags,
              make_interface_tag<
                  tmpl::_1,
                  tmpl::pin<domain::Tags::BoundaryDirectionsExterior<Dim>>,
                  tmpl::pin<get_volume_tags<FluxesComputer>>>>,
          get_volume_tags<FluxesComputer>>(
          [](const auto&... args) noexcept {
            return std::forward_as_tuple(args...);
          },
          box, direction);
      // Dispatch to derived boundary conditions class
      call_with_dynamic_type<void,
                             typename BoundaryConditions::creatable_classes>(
          &boundary_conditions,
          [&box, &direction, &fluxes_computer,
           &fluxes_args](auto* const derived_boundary_conditions) noexcept {
            using DerivedBoundaryConditions = std::decay_t<
                std::remove_pointer_t<decltype(derived_boundary_conditions)>>;
            // Get boundary conditions args out of DataBox
            const auto boundary_conditions_args = db::apply_at<
                tmpl::transform<
                    typename DerivedBoundaryConditions::argument_tags,
                    make_interface_tag<
                        tmpl::_1,
                        tmpl::pin<
                            domain::Tags::BoundaryDirectionsExterior<Dim>>,
                        tmpl::pin<get_volume_tags<DerivedBoundaryConditions>>>>,
                get_volume_tags<DerivedBoundaryConditions>>(
                [](const auto&... args) noexcept {
                  return std::forward_as_tuple(args...);
                },
                box, direction);
            // Apply the boundary conditions, mutating the buffers in the
            // DataBox
            db::mutate_apply_at<
                tmpl::list<
                    domain::Tags::Interface<
                        domain::Tags::BoundaryDirectionsExterior<Dim>,
                        db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>>,
                    domain::Tags::Interface<
                        domain::Tags::BoundaryDirectionsExterior<Dim>,
                        ::Tags::Variables<PrimalFields>>,
                    domain::Tags::Interface<
                        domain::Tags::BoundaryDirectionsExterior<Dim>,
                        ::Tags::Variables<db::wrap_tags_in<
                            ::Tags::Flux, AuxiliaryFields, tmpl::size_t<Dim>,
                            Frame::Inertial>>>>,
                tmpl::list<
                    domain::Tags::Interface<
                        domain::Tags::BoundaryDirectionsInterior<Dim>,
                        FieldsTag>,
                    domain::Tags::Interface<
                        domain::Tags::BoundaryDirectionsInterior<Dim>,
                        db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>>,
                    domain::Tags::Interface<
                        domain::Tags::BoundaryDirectionsExterior<Dim>,
                        ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                            Dim, Frame::Inertial>>>>,
                tmpl::list<>>(
                [&derived_boundary_conditions, &boundary_conditions_args,
                 &fluxes_computer,
                 &fluxes_args](const auto exterior_n_dot_fluxes,
                               const auto primal_vars_buffer,
                               const auto auxiliary_fluxes_buffer,
                               const auto& interior_vars,
                               const auto& interior_n_dot_fluxes,
                               const auto& exterior_face_normal) noexcept {
                  elliptic::impose_first_order_boundary_conditions(
                      exterior_n_dot_fluxes, primal_vars_buffer,
                      auxiliary_fluxes_buffer, interior_vars,
                      interior_n_dot_fluxes, exterior_face_normal,
                      *derived_boundary_conditions, boundary_conditions_args,
                      fluxes_computer, fluxes_args);
                },
                make_not_null(&box), direction);
          });  // call_with_dynamic_type
    }          // loop external directions

    return {std::move(box)};
  }
};

}  // namespace elliptic::dg::Actions
