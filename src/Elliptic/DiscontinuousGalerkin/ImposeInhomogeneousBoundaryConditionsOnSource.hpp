// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/SurfaceJacobian.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TupleSlice.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Adds boundary contributions to the sources
 *
 * Imposes inhomogeneous boundary conditions by adding contributions to the
 * sources of the elliptic equations. With the source modified, we may then
 * assume homogeneous boundary conditions throughout the elliptic solve.
 *
 * In this context "inhomogeneous" means the constant, variable-independent and
 * therefore nonlinear part of the boundary conditions. For example, standard
 * non-zero Dirichlet or Neumann boundary conditions are such contributions.
 */
template <typename Metavariables>
struct ImposeInhomogeneousBoundaryConditionsOnSource {
  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;
  using fields_tag = typename system::fields_tag;
  using primal_fields = typename system::primal_fields;
  using auxiliary_fields = typename system::auxiliary_fields;

  using FluxesType = typename system::fluxes;
  using fluxes_computer_tag = elliptic::Tags::FluxesComputer<FluxesType>;
  using boundary_conditions_tag =
      typename Metavariables::boundary_conditions_tag;
  using BoundaryConditions = typename boundary_conditions_tag::type;

  using fixed_sources_tag =
      db::add_tag_prefix<::Tags::FixedSource, typename system::fields_tag>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Compute the contribution to the fixed sources on all external boundaries
    auto boundary_contributions = interface_apply<
        domain::Tags::BoundaryDirectionsExterior<volume_dim>,
        tmpl::push_front<
            tmpl::append<typename FluxesType::argument_tags,
                         typename BoundaryConditions::argument_tags>,
            domain::Tags::Mesh<volume_dim>, domain::Tags::Mesh<volume_dim - 1>,
            domain::Tags::Direction<volume_dim>,
            elliptic::Tags::BoundaryConditions<typename system::primal_fields>,
            ::Tags::Normalized<
                domain::Tags::UnnormalizedFaceNormal<volume_dim>>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<volume_dim>>,
            domain::Tags::SurfaceJacobian<Frame::Logical, Frame::Inertial>,
            fluxes_computer_tag, boundary_conditions_tag,
            typename Metavariables::normal_dot_numerical_flux>,
        tmpl::push_front<tmpl::append<get_volume_tags<FluxesType>,
                                      get_volume_tags<BoundaryConditions>>,
                         domain::Tags::Mesh<volume_dim>, fluxes_computer_tag,
                         boundary_conditions_tag,
                         typename Metavariables::normal_dot_numerical_flux>>(
        [](const Mesh<volume_dim>& volume_mesh,
           const Mesh<volume_dim - 1>& face_mesh,
           const Direction<volume_dim>& direction,
           const auto& boundary_condition_types,
           const tnsr::i<DataVector, volume_dim>& exterior_face_normal,
           const Scalar<DataVector>& magnitude_of_face_normal,
           const Scalar<DataVector>& surface_jacobian,
           const FluxesType& fluxes_computer,
           const BoundaryConditions& boundary_conditions,
           const auto& numerical_fluxes_computer,
           const auto&... fluxes_and_bc_args_expanded) {
          const auto fluxes_and_bc_args =
              std::make_tuple(fluxes_and_bc_args_expanded...);
          const auto fluxes_args =
              tuple_head<tmpl::size<typename FluxesType::argument_tags>::value>(
                  fluxes_and_bc_args);
          const auto boundary_conditions_args = tuple_tail<
              tmpl::size<typename BoundaryConditions::argument_tags>::value>(
              fluxes_and_bc_args);
          const size_t dimension = direction.dimension();
          const size_t num_points = face_mesh.number_of_grid_points();
          // We get the exterior face normal out of the box, so we flip its sign
          // to obtain the interior face normal
          auto interior_face_normal = exterior_face_normal;
          for (size_t d = 0; d < volume_dim; d++) {
            interior_face_normal.get(d) *= -1.;
          }
          // Feed zero variables through the boundary conditions to retrieve
          // their nonlinear (i.e. constant) part.
          typename db::add_tag_prefix<::Tags::NormalDotFlux, fields_tag>::type
              n_dot_fluxes_exterior{num_points};
          const auto vars_interior =
              make_with_value<typename fields_tag::type>(num_points, 0.);
          const auto n_dot_fluxes_interior =
              make_with_value<typename db::add_tag_prefix<::Tags::NormalDotFlux,
                                                          fields_tag>::type>(
                  num_points, 0.);
          impose_boundary_conditions<primal_fields, auxiliary_fields>(
              make_not_null(&n_dot_fluxes_exterior), vars_interior,
              n_dot_fluxes_interior, exterior_face_normal,
              boundary_condition_types, boundary_conditions,
              boundary_conditions_args, fluxes_computer, fluxes_args);
          // Feed through numerical flux
          using NumericalFlux =
              std::decay_t<decltype(numerical_fluxes_computer)>;
          using BoundaryData = ::dg::SimpleBoundaryData<
              typename NumericalFlux::package_field_tags,
              typename NumericalFlux::package_extra_tags>;
          BoundaryData boundary_data_interior{num_points};
          ::apply<tmpl::append<
              db::wrap_tags_in<::Tags::NormalDotFlux, auxiliary_fields>,
              db::wrap_tags_in<::Tags::NormalDotFlux, primal_fields>>>(
              [&](const auto&... n_dot_fluxes) noexcept {
                std::apply(
                    [&](const auto&... expanded_fluxes_args) noexcept {
                      ::dg::NumericalFluxes::package_data(
                          make_not_null(&boundary_data_interior),
                          numerical_fluxes_computer, volume_mesh,
                          direction.opposite(), magnitude_of_face_normal,
                          n_dot_fluxes..., interior_face_normal,
                          fluxes_computer, expanded_fluxes_args...);
                    },
                    fluxes_args);
              },
              n_dot_fluxes_interior);
          BoundaryData boundary_data_exterior{num_points};
          ::apply<tmpl::append<
              db::wrap_tags_in<::Tags::NormalDotFlux, auxiliary_fields>,
              db::wrap_tags_in<::Tags::NormalDotFlux, primal_fields>>>(
              [&](const auto&... n_dot_fluxes) noexcept {
                std::apply(
                    [&](const auto&... expanded_fluxes_args) noexcept {
                      ::dg::NumericalFluxes::package_data(
                          make_not_null(&boundary_data_exterior),
                          numerical_fluxes_computer, volume_mesh, direction,
                          magnitude_of_face_normal, n_dot_fluxes...,
                          exterior_face_normal, fluxes_computer,
                          expanded_fluxes_args...);
                    },
                    fluxes_args);
              },
              n_dot_fluxes_exterior);
          typename db::add_tag_prefix<::Tags::NormalDotNumericalFlux,
                                      fixed_sources_tag>::type
              boundary_normal_dot_numerical_fluxes{num_points, 0.};
          ::dg::NumericalFluxes::normal_dot_numerical_fluxes(
              make_not_null(&boundary_normal_dot_numerical_fluxes),
              numerical_fluxes_computer, boundary_data_interior,
              boundary_data_exterior);
          // Flip sign of the boundary contributions, making them
          // contributions to the source
          if constexpr (Metavariables::massive_operator) {
            return typename fixed_sources_tag::type{
                -1. * ::dg::lift_flux_massive_no_mass_lumping(
                          std::move(boundary_normal_dot_numerical_fluxes),
                          face_mesh, surface_jacobian)};
          } else {
            return typename fixed_sources_tag::type{
                -1. *
                ::dg::lift_flux(std::move(boundary_normal_dot_numerical_fluxes),
                                volume_mesh.extents(dimension),
                                magnitude_of_face_normal)};
          }
        },
        box);

    // Add the boundary contributions to the fixed sources
    db::mutate<fixed_sources_tag>(
        make_not_null(&box),
        [&boundary_contributions](
            const gsl::not_null<typename fixed_sources_tag::type*>
                fixed_sources,
            const Mesh<volume_dim>& mesh,
            const std::unordered_set<::Direction<volume_dim>>&
                directions) noexcept {
          for (const auto& direction : directions) {
            add_slice_to_data(fixed_sources,
                              std::move(boundary_contributions.at(direction)),
                              mesh.extents(), direction.dimension(),
                              index_to_slice_at(mesh.extents(), direction));
          }
        },
        get<domain::Tags::Mesh<volume_dim>>(box),
        get<domain::Tags::BoundaryDirectionsInterior<volume_dim>>(box));

    return {std::move(box)};
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
