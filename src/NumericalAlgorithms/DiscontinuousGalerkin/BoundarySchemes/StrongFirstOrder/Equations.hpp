// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace BoundarySchemes {

// Compute the boundary flux contribution and lift it to the volume.
//
// This computes the numerical flux minus the flux dotted into the interface
// normal, projects the result to the face mesh if necessary, and then lifts it
// to the volume (still presented only on the face mesh as all other points are
// zero).
//
// Projection must happen after the numerical flux calculation so that the
// elements on either side of the mortar calculate the same result.  Projection
// must happen before flux lifting because we want the factor of the magnitude
// of the unit normal added during the lift to cancel the Jacobian factor in
// integrals to preserve conservation; this only happens if the two operations
// are done on the same grid.
template <typename VariablesTagsList, size_t FaceDim, typename BoundaryData,
          typename NumericalFluxComputer>
Variables<VariablesTagsList> strong_first_order_boundary_flux(
    const BoundaryData& local_boundary_data,
    const BoundaryData& remote_boundary_data,
    const NumericalFluxComputer& numerical_flux_computer,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const size_t extent_perpendicular_to_boundary,
    const Mesh<FaceDim>& face_mesh, const Mesh<FaceDim>& mortar_mesh,
    const std::array<Spectral::MortarSize, FaceDim>& mortar_sizes) noexcept {
  using variables_tags = VariablesTagsList;

  // Compute the Tags::NormalDotNumericalFlux from local and remote data
  Variables<db::wrap_tags_in<::Tags::NormalDotNumericalFlux, variables_tags>>
      normal_dot_numerical_fluxes{mortar_mesh.number_of_grid_points()};
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      make_not_null(&normal_dot_numerical_fluxes), numerical_flux_computer,
      local_boundary_data, remote_boundary_data);

  // Subtract the local Tags::NormalDotFlux
  tmpl::for_each<variables_tags>([&normal_dot_numerical_fluxes,
                                  &local_boundary_data](
                                     const auto tag_v) noexcept {
    using tag = tmpl::type_from<decltype(tag_v)>;
    auto& numerical_flux =
        get<::Tags::NormalDotNumericalFlux<tag>>(normal_dot_numerical_fluxes);
    const auto& local_flux =
        get<::Tags::NormalDotFlux<tag>>(local_boundary_data.field_data);
    for (size_t i = 0; i < numerical_flux.size(); ++i) {
      numerical_flux[i] -= local_flux[i];
    }
  });

  // TODO: do this elsewhere
  // Check if we need to project from the mortar back to the face
  const bool refining =
      face_mesh != mortar_mesh or
      std::any_of(mortar_sizes.begin(), mortar_sizes.end(),
                  [](const Spectral::MortarSize mortar_size) noexcept {
                    return mortar_size != Spectral::MortarSize::Full;
                  });

  // Lift flux to the volume. We still only need to provide it on the face
  // because it is zero everywhere else.
  return lift_flux(
      // Project from the mortar back to the face if needed
      refining ? project_from_mortar(normal_dot_numerical_fluxes, face_mesh,
                                     mortar_mesh, mortar_sizes)
               : std::move(normal_dot_numerical_fluxes),
      extent_perpendicular_to_boundary, magnitude_of_face_normal);
}

}  // namespace BoundarySchemes
}  // namespace dg
