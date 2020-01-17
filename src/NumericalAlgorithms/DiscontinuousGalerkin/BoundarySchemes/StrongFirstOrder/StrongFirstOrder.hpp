// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace BoundarySchemes {

namespace StrongFirstOrder_detail {

// Interface-invokable that packages the data needed to compute the boundary
// flux contribution. Combines the data for the numerical flux with the
// normal-dot-fluxes.
template <size_t Dim, typename BoundaryData, typename NormalDotFluxesTag,
          typename MagnitudeOfFaceNormalTag, typename NumericalFluxComputerTag,
          typename NumericalFluxComputer =
              db::item_type<NumericalFluxComputerTag>,
          typename ArgsTagsList = typename NumericalFluxComputer::argument_tags,
          typename VolumeArgsTagsList = get_volume_tags<NumericalFluxComputer>,
          typename PackageFieldTagsList =
              typename NumericalFluxComputer::package_field_tags,
          typename PackageExtraTagsList =
              typename NumericalFluxComputer::package_extra_tags>
struct boundary_data_computer_impl;

template <size_t Dim, typename BoundaryData, typename NormalDotFluxesTag,
          typename MagnitudeOfFaceNormalTag, typename NumericalFluxComputerTag,
          typename NumericalFluxComputer, typename... ArgsTags,
          typename... VolumeArgsTags, typename... PackageFieldTags,
          typename... PackageExtraTags>
struct boundary_data_computer_impl<
    Dim, BoundaryData, NormalDotFluxesTag, MagnitudeOfFaceNormalTag,
    NumericalFluxComputerTag, NumericalFluxComputer, tmpl::list<ArgsTags...>,
    tmpl::list<VolumeArgsTags...>, tmpl::list<PackageFieldTags...>,
    tmpl::list<PackageExtraTags...>> {
  using argument_tags = tmpl::list<
      NumericalFluxComputerTag, ::Tags::Mesh<Dim - 1>, NormalDotFluxesTag,
      ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>, ArgsTags...>;
  using volume_tags = tmpl::list<NumericalFluxComputerTag, VolumeArgsTags...>;
  static auto apply(
      const db::const_item_type<NumericalFluxComputerTag>&
          numerical_flux_computer,
      const Mesh<Dim - 1>& face_mesh,
      const db::const_item_type<NormalDotFluxesTag>& normal_dot_fluxes,
      const db::const_item_type<
          ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>&
          magnitude_of_face_normal,
      const db::const_item_type<ArgsTags>&... args) noexcept {
    BoundaryData boundary_data{face_mesh.number_of_grid_points()};
    boundary_data.field_data.assign_subset(normal_dot_fluxes);
    dg::NumericalFluxes::package_data(make_not_null(&boundary_data),
                                      numerical_flux_computer, args...);
    get<MagnitudeOfFaceNormalTag>(boundary_data.extra_data) =
        magnitude_of_face_normal;
    return boundary_data;
  }
};

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
template <typename VariablesTag, typename BoundaryData,
          typename NormalDotNumericalFluxComputer, size_t Dim>
db::const_item_type<VariablesTag> compute_boundary_flux_contribution(
    const BoundaryData& local_boundary_data,
    const BoundaryData& remote_boundary_data,
    const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const size_t extent_perpendicular_to_boundary, const Mesh<Dim>& face_mesh,
    const Mesh<Dim>& mortar_mesh,
    const std::array<Spectral::MortarSize, Dim>& mortar_sizes) noexcept {
  // Compute the Tags::NormalDotNumericalFlux from local and remote data
  db::const_item_type<
      db::add_tag_prefix<::Tags::NormalDotNumericalFlux, VariablesTag>>
      normal_dot_numerical_fluxes{mortar_mesh.number_of_grid_points()};
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      make_not_null(&normal_dot_numerical_fluxes),
      normal_dot_numerical_flux_computer, local_boundary_data,
      remote_boundary_data);

  // Subtract the local Tags::NormalDotFlux
  tmpl::for_each<db::get_variables_tags_list<VariablesTag>>([
    &normal_dot_numerical_fluxes, &local_boundary_data
  ](const auto tag_v) noexcept {
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

}  // namespace StrongFirstOrder_detail

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Boundary contributions for a strong first-order DG scheme.
 *
 * \details Computes Eq. 2.20 in \cite Teukolsky2015ega and lifts it to the
 * volume (see `dg::lift_flux`) on all mortars that touch an element. The
 * resulting boundary contributions are added to the DG operator data in
 * `db::add_tag_prefix<TemporalIdTag::template step_prefix, VariablesTag>`.
 */
template <size_t Dim, typename VariablesTag, typename NumericalFluxComputerTag,
          typename TemporalIdTag>
struct StrongFirstOrder {
  static constexpr size_t volume_dim = Dim;
  using temporal_id_tag = TemporalIdTag;
  using variables_tag = VariablesTag;
  using dt_variables_tag =
      db::add_tag_prefix<TemporalIdTag::template step_prefix, variables_tag>;
  using normal_dot_numerical_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalDotNumericalFlux, VariablesTag>;
  using normal_dot_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalDotFlux, VariablesTag>;
  using magnitude_of_face_normal_tag =
      ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<volume_dim>>;

  using BoundaryData = dg::SimpleBoundaryData<
      tmpl::remove_duplicates<
          tmpl::append<db::get_variables_tags_list<normal_dot_fluxes_tag>,
                       typename db::item_type<
                           NumericalFluxComputerTag>::package_field_tags>>,
      tmpl::append<
          typename db::item_type<NumericalFluxComputerTag>::package_extra_tags,
          // Including the magnitude of the face normal here means it will
          // be (unnecessarily) communicated to neighbors, but we don't have
          // to deal with retrieving it from element faces in the `apply`
          // function below. There, we would need to decide if we need it on
          // external faces, in addition to internal faces, instead of just
          // looping over mortars.
          tmpl::list<magnitude_of_face_normal_tag>>>;
  using boundary_data_computer =
      StrongFirstOrder_detail::boundary_data_computer_impl<
          Dim, BoundaryData, normal_dot_fluxes_tag,
          magnitude_of_face_normal_tag, NumericalFluxComputerTag>;

  /// The combined local and remote data on the mortar that is assembled
  /// during flux communication. Must have `local_insert` and `remote_insert`
  /// functions that each take `BoundaryData`. Both functions will be called
  /// exactly once in each cycle of `SendDataForFluxes` and
  /// `ReceiveDataForFluxes`.
  using mortar_data_tag =
      Tags::SimpleMortarData<db::const_item_type<TemporalIdTag>, BoundaryData>;

  using return_tags =
      tmpl::list<dt_variables_tag, ::Tags::Mortars<mortar_data_tag, Dim>>;
  using argument_tags =
      tmpl::list<::Tags::Mesh<Dim>, ::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
                 NumericalFluxComputerTag>;

  static void apply(
      const gsl::not_null<db::item_type<dt_variables_tag>*> dt_variables,
      const gsl::not_null<db::item_type<::Tags::Mortars<mortar_data_tag, Dim>>*>
          all_mortar_data,
      const Mesh<Dim>& volume_mesh,
      const db::const_item_type<::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>>&
          mortar_meshes,
      const db::const_item_type<
          ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
      const db::const_item_type<NumericalFluxComputerTag>&
          normal_dot_numerical_flux_computer) noexcept {
    // Iterate over all mortars
    for (auto& mortar_id_and_data : *all_mortar_data) {
      // Retrieve mortar data
      const auto& mortar_id = mortar_id_and_data.first;
      auto& mortar_data = mortar_id_and_data.second;
      const auto& direction = mortar_id.first;
      const size_t dimension = direction.dimension();

      // Extract local and remote data
      const auto extracted_mortar_data = mortar_data.extract();
      const BoundaryData& local_data = extracted_mortar_data.first;
      const BoundaryData& remote_data = extracted_mortar_data.second;

      db::item_type<dt_variables_tag> lifted_data(
          StrongFirstOrder_detail::compute_boundary_flux_contribution<
              variables_tag>(
              local_data, remote_data, normal_dot_numerical_flux_computer,
              get<magnitude_of_face_normal_tag>(local_data.extra_data),
              volume_mesh.extents(dimension), volume_mesh.slice_away(dimension),
              mortar_meshes.at(mortar_id), mortar_sizes.at(mortar_id)));

      // Add the flux contribution to the volume data
      add_slice_to_data(dt_variables, lifted_data, volume_mesh.extents(),
                        dimension,
                        index_to_slice_at(volume_mesh.extents(), direction));
    }
  }
};

}  // namespace BoundarySchemes
}  // namespace dg
