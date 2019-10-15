// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarAndFaceData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace BoundarySchemes {

namespace StrongFirstOrder_detail {

// Interface-invokable that packages the data needed by the numerical flux
template <size_t Dim, typename RemoteData, typename NumericalFluxComputerTag,
          typename ArgsTagsList, typename VolumeArgsTagsList>
struct remote_data_computer_impl;

template <size_t Dim, typename RemoteData, typename NumericalFluxComputerTag,
          typename... ArgsTags, typename... VolumeArgsTags>
struct remote_data_computer_impl<Dim, RemoteData, NumericalFluxComputerTag,
                                 tmpl::list<ArgsTags...>,
                                 tmpl::list<VolumeArgsTags...>> {
  using argument_tags =
      tmpl::list<NumericalFluxComputerTag, ::Tags::Mesh<Dim - 1>, ArgsTags...>;
  using volume_tags = tmpl::list<NumericalFluxComputerTag, VolumeArgsTags...>;
  static auto apply(
      const db::item_type<NumericalFluxComputerTag>& numerical_flux_computer,
      const Mesh<Dim - 1>& face_mesh,
      const db::item_type<ArgsTags>&... args) noexcept {
    RemoteData remote_data{};
    remote_data.mortar_data.initialize(face_mesh.number_of_grid_points());
    numerical_flux_computer.package_data(
        make_not_null(&(remote_data.mortar_data)), args...);
    return remote_data;
  }
};

// Interface-invokable that packages the data needed to compute the boundary
// flux contribution. Combines the data for the numerical flux with the
// normal-dot-fluxes.
template <size_t Dim, typename LocalData, typename RemoteData,
          typename NormalDotFluxesTag, typename MagnitudeOfFaceNormalTag>
struct local_data_computer_impl {
  using argument_tags =
      tmpl::list<::Tags::Direction<Dim>, NormalDotFluxesTag,
                 ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>;
  static auto apply(const Direction<Dim>& direction,
                    const db::item_type<NormalDotFluxesTag>& normal_dot_fluxes,
                    const db::item_type<
                        ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>&
                        magnitude_of_face_normal,
                    const DirectionMap<Dim, RemoteData>& remote_data) noexcept {
    LocalData local_data{};
    local_data.mortar_data.initialize(
        normal_dot_fluxes.number_of_grid_points());
    local_data.mortar_data.assign_subset(normal_dot_fluxes);
    local_data.mortar_data.assign_subset(remote_data.at(direction).mortar_data);
    get<MagnitudeOfFaceNormalTag>(local_data.face_data) =
        magnitude_of_face_normal;
    return local_data;
  }
};

// Helper function to unpack arguments when invoking the numerical flux computer
template <typename NormalDotNumericalFluxComputer,
          typename... NumericalFluxTags, typename... SelfTags,
          typename... PackagedTags>
void apply_normal_dot_numerical_flux(
    const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
        numerical_fluxes,
    const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
    const Variables<tmpl::list<SelfTags...>>& self_packaged_data,
    const Variables<tmpl::list<PackagedTags...>>&
        neighbor_packaged_data) noexcept {
  normal_dot_numerical_flux_computer(
      make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
      get<PackagedTags>(self_packaged_data)...,
      get<PackagedTags>(neighbor_packaged_data)...);
}

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
template <typename VariablesTag, typename LocalMortarDataTags,
          typename RemoteMortarDataTags,
          typename NormalDotNumericalFluxComputer, size_t Dim>
db::const_item_type<VariablesTag> compute_boundary_flux_contribution(
    const Variables<LocalMortarDataTags>& local_mortar_data,
    const Variables<RemoteMortarDataTags>& remote_mortar_data,
    const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const size_t extent_perpendicular_to_boundary, const Mesh<Dim>& face_mesh,
    const Mesh<Dim>& mortar_mesh,
    const std::array<Spectral::MortarSize, Dim>& mortar_sizes) noexcept {
  // Compute the Tags::NormalDotNumericalFlux from local and remote data
  db::const_item_type<
      db::add_tag_prefix<::Tags::NormalDotNumericalFlux, VariablesTag>>
      normal_dot_numerical_fluxes(mortar_mesh.number_of_grid_points(), 0.0);
  apply_normal_dot_numerical_flux(make_not_null(&normal_dot_numerical_fluxes),
                                  normal_dot_numerical_flux_computer,
                                  local_mortar_data, remote_mortar_data);

  // Subtract the local Tags::NormalDotFlux
  tmpl::for_each<db::get_variables_tags_list<VariablesTag>>([
    &normal_dot_numerical_fluxes, &local_mortar_data
  ](const auto tag_v) noexcept {
    using tag = tmpl::type_from<decltype(tag_v)>;
    auto& numerical_flux =
        get<::Tags::NormalDotNumericalFlux<tag>>(normal_dot_numerical_fluxes);
    const auto& local_flux = get<::Tags::NormalDotFlux<tag>>(local_mortar_data);
    for (size_t i = 0; i < numerical_flux.size(); ++i) {
      numerical_flux[i] -= local_flux[i];
    }
  });

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
 * \brief Strong
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

  /// Data that is communicated to and from the remote element.
  using RemoteData = dg::MortarAndFaceData<
      typename db::item_type<NumericalFluxComputerTag>::package_tags>;
  using remote_data_computer =
      StrongFirstOrder_detail::remote_data_computer_impl<
          Dim, RemoteData, NumericalFluxComputerTag,
          typename db::item_type<NumericalFluxComputerTag>::argument_tags,
          get_volume_tags<typename db::item_type<NumericalFluxComputerTag>>>;

  /// Data from the local element that is needed to compute the lifted boundary
  /// flux contributions
  using LocalData = dg::MortarAndFaceData<
      tmpl::remove_duplicates<
          tmpl::append<db::get_variables_tags_list<normal_dot_fluxes_tag>,
                       typename RemoteData::mortar_tags>>,
      tmpl::list<magnitude_of_face_normal_tag>>;
  using local_data_computer = StrongFirstOrder_detail::local_data_computer_impl<
      Dim, LocalData, RemoteData, normal_dot_fluxes_tag,
      magnitude_of_face_normal_tag>;

  /// The combined local and remote data on the mortar that is assembled
  /// during flux communication. Must have `local_insert` and `remote_insert`
  /// functions that take `LocalData` and `RemoteData`, respectively. Both
  /// functions will be called exactly once in each cycle of
  /// `SendDataForFluxes` and `ReceiveDataForFluxes`.
  using mortar_data_tag =
      Tags::SimpleBoundaryData<db::const_item_type<TemporalIdTag>, LocalData,
                               RemoteData>;

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
      const LocalData& local_data = extracted_mortar_data.first;
      const RemoteData& remote_data = extracted_mortar_data.second;

      db::item_type<dt_variables_tag> lifted_data(
          StrongFirstOrder_detail::compute_boundary_flux_contribution<
              variables_tag>(
              local_data.mortar_data, remote_data.mortar_data,
              normal_dot_numerical_flux_computer,
              get<magnitude_of_face_normal_tag>(local_data.face_data),
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
