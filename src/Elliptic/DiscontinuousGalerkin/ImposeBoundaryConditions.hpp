// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Packages data on external boundaries so that they represent
 * homogeneous (zero) Dirichlet boundary conditions.
 *
 * This action imposes homogeneous boundary conditions on all fields in
 * `system::impose_boundary_conditions_on_fields`. The fields are wrapped in
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
 * - Metavariables:
 *   - `normal_dot_numerical_flux`
 *   - `temporal_id`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *   - `impose_boundary_conditions_on_fields`
 * - ConstGlobalCache:
 *   - `normal_dot_numerical_flux`
 * - DataBox:
 *   - `Tags::Element<volume_dim>`
 *   - `temporal_id`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `Tags::BoundaryDirectionsExterior<volume_dim>`
 *   - `interior<variables_tag>`
 *   - `exterior<variables_tag>`
 *   - `interior<normal_dot_numerical_flux::type::argument_tags>`
 *   - `exterior<normal_dot_numerical_flux::type::argument_tags>`
 *
 * DataBox changes:
 * - Modifies:
 *   - `exterior<variables_tag>`
 *   - `Tags::VariablesBoundaryData`
 */
template <typename FluxLiftingScheme, typename DirichletFieldsTagsList>
struct ImposeHomogeneousDirichletBoundaryConditions {
 private:
  static constexpr size_t volume_dim = FluxLiftingScheme::volume_dim;

  template <typename F, typename DirectionsTag, typename DataBoxType,
            typename... ArgsTags, typename... ExtraArgs>
  static auto compute_packaged_face_data(
      const DataBoxType& box, const Direction<volume_dim>& direction,
      tmpl::list<ArgsTags...> /*meta*/,
      const ExtraArgs&... extra_args) noexcept {
    return F::apply(
        get<::Tags::Interface<DirectionsTag, ArgsTags>>(box).at(direction)...,
        extra_args...);
  }

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using dirichlet_tags = DirichletFieldsTagsList;

    // Set the data on exterior (ghost) faces to impose the boundary conditions
    db::mutate<::Tags::Interface<::Tags::BoundaryDirectionsExterior<volume_dim>,
                                 typename FluxLiftingScheme::variables_tag>>(
        make_not_null(&box),
        // Need to use FluxLiftingScheme::volume_dim below instead of just
        // volume_dim to avoid an ICE on gcc 7.
        [](const gsl::not_null<db::item_type<
               ::Tags::Interface<::Tags::BoundaryDirectionsExterior<
                                     FluxLiftingScheme::volume_dim>,
                                 typename FluxLiftingScheme::variables_tag>>*>
               exterior_boundary_vars,
           const db::item_type<
               ::Tags::Interface<::Tags::BoundaryDirectionsInterior<
                                     FluxLiftingScheme::volume_dim>,
                                 typename FluxLiftingScheme::variables_tag>>&
               interior_vars) noexcept {
          for (auto& exterior_direction_and_vars : *exterior_boundary_vars) {
            auto& direction = exterior_direction_and_vars.first;
            auto& exterior_vars = exterior_direction_and_vars.second;

            // By default, use the variables on the external boundary for the
            // exterior
            exterior_vars = interior_vars.at(direction);

            // For those variables where we have boundary conditions, impose
            // zero Dirichlet b.c. here. The non-zero boundary conditions are
            // handled as contributions to the source in InitializeElement.
            // Imposing them here would not work because when are working with
            // the linear solver operand.
            tmpl::for_each<dirichlet_tags>([
              &interior_vars, &exterior_vars, &direction
            ](auto dirichlet_tag_val) noexcept {
              using dirichlet_tag =
                  tmpl::type_from<decltype(dirichlet_tag_val)>;
              // Use mirror principle. This only works for scalars right now.
              get(get<dirichlet_tag>(exterior_vars)) =
                  -1. * get<dirichlet_tag>(interior_vars.at(direction)).get();
            });
          }
        },
        get<::Tags::Interface<::Tags::BoundaryDirectionsInterior<volume_dim>,
                              typename FluxLiftingScheme::variables_tag>>(box));

    // Store local and packaged data on the mortars
    for (const auto& direction :
         db::get<::Tags::Element<volume_dim>>(box).external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());

      // HACK until we can retrieve cache tags from the DataBox in compute items
      // No need for projections since the ghost mortar always covers the full
      // face
      auto remote_data_interior = compute_packaged_face_data<
          typename FluxLiftingScheme::package_remote_data,
          ::Tags::BoundaryDirectionsInterior<volume_dim>>(
          box, direction,
          typename FluxLiftingScheme::package_remote_data::argument_tags{},
          get<typename FluxLiftingScheme::numerical_flux_computer_tag>(cache));
      auto local_data_interior = compute_packaged_face_data<
          typename FluxLiftingScheme::package_local_data,
          ::Tags::BoundaryDirectionsInterior<volume_dim>>(
          box, direction,
          typename FluxLiftingScheme::package_local_data::argument_tags{},
          std::move(remote_data_interior));
      auto remote_data_exterior = compute_packaged_face_data<
          typename FluxLiftingScheme::package_remote_data,
          ::Tags::BoundaryDirectionsExterior<volume_dim>>(
          box, direction,
          typename FluxLiftingScheme::package_remote_data::argument_tags{},
          get<typename FluxLiftingScheme::numerical_flux_computer_tag>(cache));

      using all_mortar_data_tag =
          ::Tags::Mortars<typename FluxLiftingScheme::mortar_data_tag,
                          volume_dim>;
      db::mutate<all_mortar_data_tag>(
          make_not_null(&box),
          [&mortar_id, &local_data_interior, &remote_data_exterior ](
              const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                  all_mortar_data,
              const db::item_type<typename FluxLiftingScheme::temporal_id_tag>&
                  temporal_id) noexcept {
            all_mortar_data->at(mortar_id).local_insert(
                temporal_id, std::move(local_data_interior));
            all_mortar_data->at(mortar_id).remote_insert(
                temporal_id, std::move(remote_data_exterior));
          },
          db::get<typename FluxLiftingScheme::temporal_id_tag>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace dg
}  // namespace Elliptic
