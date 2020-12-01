// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace elliptic::dg {

/// Package all data needed by the numerical flux computer on one side of the
/// face
template <size_t Dim, typename NumericalFluxesComputerType,
          typename FluxesComputerType, typename NormalDotFluxesTags,
          typename DivFluxesTags, typename... FluxesArgs,
          typename... AuxiliaryFields>
auto package_boundary_data(
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const FluxesComputerType& fluxes_computer, const Mesh<Dim>& volume_mesh,
    const Direction<Dim>& direction, const Mesh<Dim - 1>& face_mesh,
    const tnsr::i<DataVector, Dim>& face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const Variables<NormalDotFluxesTags>& n_dot_fluxes,
    const Variables<DivFluxesTags>& div_fluxes_on_face,
    const std::tuple<FluxesArgs...>& fluxes_args,
    tmpl::list<AuxiliaryFields...> /*meta*/) noexcept {
  return std::apply(
      [&](const auto&... expanded_fluxes_args) {
        return ::dg::FirstOrderScheme::package_boundary_data(
            numerical_fluxes_computer, face_mesh, n_dot_fluxes,
            // The following arguments are passed on to the numerical flux
            // computer. This currently assumes we're dealing with the internal
            // penalty flux, but could be generalized.
            volume_mesh, direction, face_normal_magnitude,
            get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_fluxes)...,
            get<::Tags::div<::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>,
                                         Frame::Inertial>>>(
                div_fluxes_on_face)...,
            face_normal, fluxes_computer, expanded_fluxes_args...);
      },
      fluxes_args);
}

/// On exterior ("ghost") faces, manufacture boundary data that represent
/// homogeneous Dirichlet boundary conditions
template <typename PrimalFields, typename AuxiliaryFields, size_t Dim,
          typename BoundaryData, typename FluxesComputerType,
          typename NumericalFluxesComputerType, typename... FluxesArgs,
          typename FieldsTags = tmpl::append<PrimalFields, AuxiliaryFields>,
          typename FluxesTags = db::wrap_tags_in<
              ::Tags::Flux, FieldsTags, tmpl::size_t<Dim>, Frame::Inertial>,
          typename DivFluxesTags = db::wrap_tags_in<::Tags::div, FluxesTags>>
void package_exterior_boundary_data(
    const gsl::not_null<BoundaryData*> boundary_data,
    const Variables<FieldsTags>& vars_on_interior_face,
    const Variables<DivFluxesTags>& div_fluxes_on_interior_face,
    const Mesh<Dim>& volume_mesh, const Direction<Dim>& interior_direction,
    const Mesh<Dim - 1>& face_mesh,
    const tnsr::i<DataVector, Dim>& interior_face_normal,
    const Scalar<DataVector>& interior_face_normal_magnitude,
    const FluxesComputerType& fluxes_computer,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const std::tuple<FluxesArgs...>& fluxes_args) noexcept {
  Variables<FieldsTags> ghost_vars{
      vars_on_interior_face.number_of_grid_points()};
  ::elliptic::dg::homogeneous_dirichlet_boundary_conditions<PrimalFields>(
      make_not_null(&ghost_vars), vars_on_interior_face);
  const auto ghost_fluxes = std::apply(
      [&ghost_vars, &fluxes_computer](const auto&... expanded_fluxes_args) {
        return ::elliptic::first_order_fluxes<Dim, PrimalFields,
                                              AuxiliaryFields>(
            ghost_vars, fluxes_computer, expanded_fluxes_args...);
      },
      fluxes_args);
  auto exterior_face_normal = interior_face_normal;
  for (size_t d = 0; d < Dim; d++) {
    exterior_face_normal.get(d) *= -1.;
  }
  const auto ghost_normal_dot_fluxes =
      normal_dot_flux<FieldsTags>(exterior_face_normal, ghost_fluxes);
  *boundary_data = elliptic::dg::package_boundary_data(
      numerical_fluxes_computer, fluxes_computer, volume_mesh,
      interior_direction.opposite(), face_mesh, exterior_face_normal,
      interior_face_normal_magnitude, ghost_normal_dot_fluxes,
      div_fluxes_on_interior_face, fluxes_args, AuxiliaryFields{});
}

/// Contribute boundary data to the operator once it is available on both sides
/// of a mortar. Typically you would invoke `package_boundary_data` on both
/// sides of the mortar, communicate the results and then call this function.
template <size_t Dim, typename FieldsTagsList,
          typename NumericalFluxesComputerType, typename BoundaryData>
void apply_boundary_contribution(
    const gsl::not_null<Variables<FieldsTagsList>*> result,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const BoundaryData& local_boundary_data,
    const BoundaryData& remote_boundary_data,
    const Scalar<DataVector>& magnitude_of_face_normal, const Mesh<Dim>& mesh,
    const Direction<Dim>& direction, const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size) noexcept {
  const size_t dimension = direction.dimension();
  auto boundary_contribution = ::dg::FirstOrderScheme::boundary_flux(
      local_boundary_data, remote_boundary_data, numerical_fluxes_computer,
      magnitude_of_face_normal, mesh.extents(dimension),
      mesh.slice_away(dimension), mortar_mesh, mortar_size);
  add_slice_to_data(result, std::move(boundary_contribution), mesh.extents(),
                    dimension, index_to_slice_at(mesh.extents(), direction));
}

}  // namespace elliptic::dg
