// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
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
template <typename BoundaryConditionsTag, typename FluxesComputerTag>
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
    const auto& boundary_conditions = get<BoundaryConditionsTag>(box);
    const auto& fluxes_computer = get<FluxesComputerTag>(box);
    const auto& numerical_fluxes_computer =
        get<NumericalFluxesComputerTag>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& volume_mesh = db::get<domain::Tags::Mesh<Dim>>(box);

    for (const auto& direction : element.external_directions()) {
      const auto face_mesh = volume_mesh.slice_away(direction.dimension());
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const auto& exterior_face_normal = db::get<domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsExterior<Dim>,
          ::Tags::Normalized<
              domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>>(box)
                                             .at(direction);
      // Allocate buffers for feeding zero variables through the boundary
      // conditions to retrieve their nonlinear (i.e. constant) part. Note that
      // we are considering linear elliptic equations here, so the only possible
      // nonlinearity in the boundary conditions is a constant part that can be
      // contributed to the fixed sources.
      typename db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>::type
          exterior_n_dot_fluxes{face_num_points};
      Variables<PrimalFields> primal_vars_buffer{face_num_points};
      Variables<db::wrap_tags_in<::Tags::Flux, PrimalFields, tmpl::size_t<Dim>,
                                 Frame::Inertial>>
          auxiliary_fluxes_buffer{face_num_points};
      const auto zero_vars =
          make_with_value<typename FieldsTag::type>(face_num_points, 0.);
      const auto zero_n_dot_fluxes = make_with_value<
          typename db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>::type>(
          face_num_points, 0.);
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
            return std::forward_as_tuple(args...)
          },
          box, direction);
      // Dispatch to derived boundary conditions class
      call_with_dynamic_type<void,
                             typename std::decay_t<decltype(
                                 boundary_conditions)>::creatable_classes>(
          &boundary_conditions,
          [&box, &direction, &exterior_n_dot_fluxes, &primal_vars_buffer,
           &auxiliary_fluxes_buffer, &zero_vars, &zero_n_dot_fluxes,
           &exterior_face_normal, &fluxes_computer,
           &fluxes_args](auto* const derived_boundary_conditions) noexcept {
            using DerivedBoundaryConditions =
                std::decay_t<decltype(derived_boundary_conditions)>;
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
                  return std::forward_as_tuple(args...)
                },
                box, direction);
            // Feed zero variables through the boundary conditions
            elliptic::impose_first_order_boundary_conditions(
                make_not_null(&exterior_n_dot_fluxes),
                make_not_null(&primal_vars_buffer),
                make_not_null(&auxiliary_fluxes_buffer), zero_vars,
                zero_n_dot_fluxes, exterior_face_normal,
                *derived_boundary_conditions, boundary_conditions_args,
                fluxes_computer, fluxes_args);
          });  // call_with_dynamic_type
      // Feed through numerical flux
      using NumericalFlux = std::decay_t<decltype(numerical_fluxes_computer)>;
      using BoundaryData =
          ::dg::SimpleBoundaryData<typename NumericalFlux::package_field_tags,
                                   typename NumericalFlux::package_extra_tags>;
      BoundaryData boundary_data_interior{face_num_points};
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
                      n_dot_fluxes..., interior_face_normal, fluxes_computer,
                      expanded_fluxes_args...);
                },
                fluxes_args);
          },
          zero_n_dot_fluxes);
      BoundaryData boundary_data_exterior{face_num_points};
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
          exterior_n_dot_fluxes);
      typename db::add_tag_prefix<::Tags::NormalDotNumericalFlux,
                                  fixed_sources_tag>::type
          boundary_normal_dot_numerical_fluxes{face_num_points, 0.};
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
      // Add the boundary contributions to the fixed sources
      db::mutate_apply_at<tmpl::list<fixed_sources_tag>, tmpl::list<>,
                          tmpl::list<fixed_sources_tag>>(
          [&](const gsl::not_null<typename fixed_sources_tag::type*>
                  fixed_sources) noexcept {
            add_slice_to_data(fixed_sources, std::move(boundary_contributions),
                              mesh.extents(), direction.dimension(),
                              index_to_slice_at(mesh.extents(), direction));
          },
          make_not_null(&box));
    }  // loop external directions

    return {std::move(box)}; 
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
