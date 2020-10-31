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
#include "Elliptic/BoundaryConditions.hpp"
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
template <typename PrimalFields, typename AuxiliaryFields, typename TagsList>
void homogeneous_boundary_conditions(
    const gsl::not_null<Variables<TagsList>*> exterior_n_dot_fluxes,
    const Variables<TagsList>& interior_n_dot_fluxes,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<elliptic::Tags::BoundaryCondition, PrimalFields>>&
        boundary_condition_types) noexcept {
  // By default, use the variables on the external boundary for the
  // exterior
  *exterior_n_dot_fluxes = interior_n_dot_fluxes;
  // For those variables where we have boundary conditions, impose zero b.c.
  // here. The non-zero boundary conditions are handled as contributions to the
  // source in InitializeElement. Imposing them here would break linearity of
  // the DG operator.
  tmpl::for_each<PrimalFields>(
      [&exterior_n_dot_fluxes, &boundary_condition_types](auto tag_v) noexcept {
        using tag = tmpl::type_from<decltype(tag_v)>;
        using aux_tag =
            tmpl::at<AuxiliaryFields, tmpl::index_of<PrimalFields, tag>>;
        switch (get<elliptic::Tags::BoundaryCondition<tag>>(
            boundary_condition_types)) {
          case elliptic::BoundaryCondition::Dirichlet: {
            auto& invert_field =
                get<::Tags::NormalDotFlux<tag>>(*exterior_n_dot_fluxes);
            for (size_t i = 0; i < invert_field.size(); i++) {
              invert_field[i] *= -1.;
            }
          } break;
          case elliptic::BoundaryCondition::Neumann: {
            auto& invert_field =
                get<::Tags::NormalDotFlux<aux_tag>>(*exterior_n_dot_fluxes);
            for (size_t i = 0; i < invert_field.size(); i++) {
              invert_field[i] *= -1.;
            }
          } break;
          default:
            ERROR("Invalid case");
        }
      });
}

namespace detail {
template <typename PrimalFields, typename AuxiliaryFields>
struct ImposeBoundaryConditionsImpl;

template <typename... PrimalFields, typename AuxiliaryFields>
struct ImposeBoundaryConditionsImpl<tmpl::list<PrimalFields...>,
                                    AuxiliaryFields> {
  template <size_t Dim, typename InteriorVars, typename InteriorFluxes,
            typename BoundaryConditions, typename... BoundaryConditionsArgs,
            typename FluxesComputer, typename... FluxesArgs>
  static void apply(
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::NormalDotFlux,
          tmpl::append<tmpl::list<PrimalFields...>, AuxiliaryFields>>>*>
          exterior_n_dot_fluxes,
      const Variables<InteriorVars>& interior_vars,
      const Variables<InteriorFluxes>& interior_n_dot_fluxes,
      const tnsr::i<DataVector, Dim>& exterior_face_normal,
      const tuples::TaggedTuple<
          elliptic::Tags::BoundaryCondition<PrimalFields>...>&
          boundary_condition_types,
      const BoundaryConditions& boundary_conditions,
      const std::tuple<BoundaryConditionsArgs...>& boundary_conditions_args,
      const FluxesComputer& fluxes_computer,
      const std::tuple<FluxesArgs...>& fluxes_args) noexcept {
    using primal_fields = tmpl::list<PrimalFields...>;
    using auxiliary_fields = AuxiliaryFields;
    // First, copy interior n.F over to the exterior to impose no
    // boundary conditions by default. We invert the sign to account for
    // the inverted normal on exterior faces.
    *exterior_n_dot_fluxes = -1. * interior_n_dot_fluxes;
    // Then, iterate primal fields to impose boundary conditions.
    // Dirichlet conditions are imposed on the n.F_v and Neumann
    // conditions are imposed on the n.F_u.
    auto dirichlet_fields = interior_vars;
    std::apply(
        [&boundary_conditions, &dirichlet_fields, &exterior_n_dot_fluxes,
         &exterior_face_normal](
            const auto&... expanded_boundary_conditions_args) noexcept {
          boundary_conditions.apply(
              make_not_null(&get<PrimalFields>(dirichlet_fields))...,
              make_not_null(&get<::Tags::NormalDotFlux<PrimalFields>>(
                  *exterior_n_dot_fluxes))...,
              exterior_face_normal, expanded_boundary_conditions_args...);
        },
        boundary_conditions_args);
    const auto dirichlet_fluxes = std::apply(
        [&dirichlet_fields,
         &fluxes_computer](const auto&... expanded_fluxes_args) noexcept {
          return elliptic::first_order_fluxes<Dim, primal_fields,
                                              auxiliary_fields>(
              dirichlet_fields, fluxes_computer, expanded_fluxes_args...);
        },
        fluxes_args);
    const auto n_dot_dirichlet_fluxes =
        normal_dot_flux<tmpl::append<primal_fields, auxiliary_fields>>(
            exterior_face_normal, dirichlet_fluxes);
    tmpl::for_each<primal_fields>([&](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      using aux_tag =
          tmpl::at<auxiliary_fields, tmpl::index_of<primal_fields, tag>>;
      switch (get<elliptic::Tags::BoundaryCondition<tag>>(
          boundary_condition_types)) {
        case elliptic::BoundaryCondition::Dirichlet: {
          // Impose Dirichlet conditions such that {{u}} = (u_int +
          // u_ext) / 2 = u_D. Just setting u_ext = u_D also works but
          // converges way slower.
          auto& exterior_n_dot_aux_flux =
              get<::Tags::NormalDotFlux<aux_tag>>(*exterior_n_dot_fluxes);
          for (size_t i = 0; i < exterior_n_dot_aux_flux.size(); i++) {
            exterior_n_dot_aux_flux[i] *= -1.;
            exterior_n_dot_aux_flux[i] +=
                2. *
                get<::Tags::NormalDotFlux<aux_tag>>(n_dot_dirichlet_fluxes)[i];
          }
        } break;
        case elliptic::BoundaryCondition::Neumann: {
          auto& exterior_n_dot_flux =
              get<::Tags::NormalDotFlux<tag>>(*exterior_n_dot_fluxes);
          for (size_t i = 0; i < exterior_n_dot_flux.size(); i++) {
            exterior_n_dot_flux[i] *= 2.;
            exterior_n_dot_flux[i] +=
                get<::Tags::NormalDotFlux<tag>>(interior_n_dot_fluxes)[i];
          }
        } break;
        default:
          ERROR("Invalid case");
      }
    });
  }
};
}  // namespace detail

template <typename PrimalFields, typename AuxiliaryFields, size_t Dim,
          typename InteriorVars, typename InteriorFluxes,
          typename BoundaryConditions, typename... BoundaryConditionsArgs,
          typename FluxesComputer, typename... FluxesArgs>
void impose_boundary_conditions(
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::NormalDotFlux, tmpl::append<PrimalFields, AuxiliaryFields>>>*>
        exterior_n_dot_fluxes,
    const Variables<InteriorVars>& interior_vars,
    const Variables<InteriorFluxes>& interior_n_dot_fluxes,
    const tnsr::i<DataVector, Dim>& exterior_face_normal,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<elliptic::Tags::BoundaryCondition, PrimalFields>>&
        boundary_condition_types,
    const BoundaryConditions& boundary_conditions,
    const std::tuple<BoundaryConditionsArgs...>& boundary_conditions_args,
    const FluxesComputer& fluxes_computer,
    const std::tuple<FluxesArgs...>& fluxes_args) noexcept {
  detail::ImposeBoundaryConditionsImpl<PrimalFields, AuxiliaryFields>::apply(
      exterior_n_dot_fluxes, interior_vars, interior_n_dot_fluxes,
      exterior_face_normal, boundary_condition_types, boundary_conditions,
      boundary_conditions_args, fluxes_computer, fluxes_args);
}

namespace Actions {

template <
    typename BoundaryConditionsTag, typename FieldsTag, typename PrimalFields,
    typename AuxiliaryFields, typename FluxesComputerTag,
    typename BcArgsTags = typename BoundaryConditionsTag::type::argument_tags,
    typename FluxesArgsTags = typename FluxesComputerTag::type::argument_tags>
struct ImposeBoundaryConditions;

template <typename BoundaryConditionsTag, typename FieldsTag,
          typename PrimalFields, typename AuxiliaryFields,
          typename FluxesComputerTag, typename... BcArgsTags,
          typename... FluxesArgsTags>
struct ImposeBoundaryConditions<BoundaryConditionsTag, FieldsTag, PrimalFields,
                                AuxiliaryFields, FluxesComputerTag,
                                tmpl::list<BcArgsTags...>,
                                tmpl::list<FluxesArgsTags...>> {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<domain::Tags::Interface<
        domain::Tags::BoundaryDirectionsExterior<Dim>,
        db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>>>(
        make_not_null(&box),
        [](const auto all_exterior_n_dot_fluxes, const auto& all_interior_vars,
           const auto& all_interior_n_dot_fluxes,
           const auto& all_exterior_face_normals,
           const auto& all_boundary_condition_types,
           const auto& boundary_conditions, const auto& fluxes_computer,
           const auto& boundary_conditions_args,
           const auto& fluxes_args) noexcept {
          for (auto& direction_and_exterior_n_dot_fluxes :
               *all_exterior_n_dot_fluxes) {
            const auto& direction = direction_and_exterior_n_dot_fluxes.first;
            auto& exterior_n_dot_fluxes =
                direction_and_exterior_n_dot_fluxes.second;
            std::apply(
                [&](const auto&... expanded_boundary_conditions_args) {
                  std::apply(
                      [&](const auto&... expanded_fluxes_args) {
                        impose_boundary_conditions<PrimalFields,
                                                   AuxiliaryFields>(
                            make_not_null(&exterior_n_dot_fluxes),
                            all_interior_vars.at(direction),
                            all_interior_n_dot_fluxes.at(direction),
                            all_exterior_face_normals.at(direction),
                            all_boundary_condition_types.at(direction),
                            boundary_conditions,
                            std::make_tuple(
                                InterfaceHelpers_detail::unmap_interface_args<
                                    tmpl::list_contains_v<
                                        get_volume_tags<
                                            typename BoundaryConditionsTag::
                                                type>,
                                        BcArgsTags>>::
                                    apply(
                                        direction,
                                        expanded_boundary_conditions_args)...),
                            fluxes_computer,
                            std::make_tuple(
                                InterfaceHelpers_detail::unmap_interface_args<
                                    tmpl::list_contains_v<
                                        get_volume_tags<
                                            typename FluxesComputerTag::type>,
                                        FluxesArgsTags>>::
                                    apply(direction, expanded_fluxes_args)...));
                      },
                      fluxes_args);
                },
                boundary_conditions_args);
          }
        },
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>, FieldsTag>>(box),
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>>>(box),
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsExterior<Dim>,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(
            box),
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsExterior<Dim>,
            elliptic::Tags::BoundaryConditions<PrimalFields>>>(box),
        db::get<BoundaryConditionsTag>(box), db::get<FluxesComputerTag>(box),
        std::make_tuple(
            db::get<typename InterfaceHelpers_detail::make_interface_tag_impl<
                BcArgsTags, domain::Tags::BoundaryDirectionsExterior<Dim>,
                get_volume_tags<typename BoundaryConditionsTag::type>>::type>(
                box)...),
        std::make_tuple(
            db::get<typename InterfaceHelpers_detail::make_interface_tag_impl<
                FluxesArgsTags, domain::Tags::BoundaryDirectionsExterior<Dim>,
                get_volume_tags<typename FluxesComputerTag::type>>::type>(
                box)...));

    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
