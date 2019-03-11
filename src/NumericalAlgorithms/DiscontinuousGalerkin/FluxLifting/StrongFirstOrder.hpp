// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function lift_flux.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename SubTags, typename Tags>
Variables<SubTags> subset_variables(const Variables<Tags>& vars) noexcept {
  Variables<SubTags> sub_vars(vars.number_of_grid_points());
  tmpl::for_each<SubTags>([&](const auto tag_v) noexcept {
    using tag = tmpl::type_from<decltype(tag_v)>;
    get<tag>(sub_vars) = get<tag>(vars);
  });
  return sub_vars;
}
}  // namespace

namespace dg {
namespace FluxLifting {

template <size_t Dim, typename VariablesTag, typename FluxComputerType,
          typename NumericalFluxComputerTag, typename TemporalIdTag>
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

  /// A `Variables` that is communicated to and from the remote element
  using RemoteData =
      Variables<typename NumericalFluxComputerTag::type::package_tags>;

  /// Data from the local element that is needed to compute the lifted fluxes
  struct LocalData {
    /// Data on the mortar mesh
    Variables<tmpl::remove_duplicates<
        tmpl::append<db::get_variables_tags_list<normal_dot_fluxes_tag>,
                     typename RemoteData::tags_list>>>
        mortar_data;
    /// Magnitude of the face normal on the face mesh
    Scalar<DataVector> magnitude_of_face_normal;
    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p) noexcept {
      p | mortar_data;
      p | magnitude_of_face_normal;
    }
  };

  // TODO: make compute items once we can retrieve cache tags through the
  // databox
  template <typename ArgsTagsList>
  struct package_remote_data_impl;
  template <typename... ArgsTags>
  struct package_remote_data_impl<tmpl::list<ArgsTags...>> {
    using argument_tags = tmpl::list<::Tags::Mesh<volume_dim - 1>, ArgsTags...>;
    static RemoteData apply(const Mesh<volume_dim - 1>& face_mesh,
                            const db::item_type<ArgsTags>&... args,
                            const typename NumericalFluxComputerTag::type&
                                numerical_flux_computer) noexcept {
      RemoteData remote_data{face_mesh.number_of_grid_points()};
      numerical_flux_computer.package_data(make_not_null(&remote_data),
                                           args...);
      return remote_data;
    }
  };
  using package_remote_data = package_remote_data_impl<
      typename NumericalFluxComputerTag::type::argument_tags>;

  struct package_local_data {
    using argument_tags = tmpl::list<
        normal_dot_fluxes_tag,
        ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<volume_dim>>>;
    static LocalData apply(
        const db::item_type<normal_dot_fluxes_tag>& normal_dot_fluxes,
        const db::item_type<
            ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<volume_dim>>>&
            magnitude_of_face_normal,
        const RemoteData& remote_data) noexcept {
      LocalData local_data{};
      local_data.mortar_data.initialize(
          normal_dot_fluxes.number_of_grid_points());
      local_data.mortar_data.assign_subset(normal_dot_fluxes);
      local_data.mortar_data.assign_subset(remote_data);
      local_data.magnitude_of_face_normal = magnitude_of_face_normal;
      return local_data;
    }
  };

  static RemoteData project_to_mortar(
      const RemoteData& face_data, const Mesh<volume_dim - 1>& face_mesh,
      const Mesh<volume_dim - 1>& mortar_mesh,
      const std::array<Spectral::MortarSize, volume_dim - 1>& mortar_size) {
    return dg::project_to_mortar(face_data, face_mesh, mortar_mesh,
                                 mortar_size);
  }

  static LocalData project_to_mortar(
      const LocalData& face_data, const Mesh<volume_dim - 1>& face_mesh,
      const Mesh<volume_dim - 1>& mortar_mesh,
      const std::array<Spectral::MortarSize, volume_dim - 1>& mortar_size) {
    LocalData projected_local_data{};
    projected_local_data.mortar_data = dg::project_to_mortar(
        face_data.mortar_data, face_mesh, mortar_mesh, mortar_size);
    projected_local_data.magnitude_of_face_normal =
        face_data.magnitude_of_face_normal;
    return projected_local_data;
  }

  /// The combined local and remote data on the mortar that is assembled during
  /// flux communication. Must have `local_insert` and `remote_insert` functions
  /// that take `LocalData` and `RemoteData`, respectively. Both functions will
  /// be called exactly once in each cycle of `SendDataForFluxes` and
  /// `ReceiveDataForFluxes`.
  using mortar_data_tag = Tags::SimpleBoundaryData<db::item_type<TemporalIdTag>,
                                                   LocalData, RemoteData>;

  using return_tags = tmpl::list<dt_variables_tag,
                                 ::Tags::Mortars<mortar_data_tag, volume_dim>>;
  using argument_tags = tmpl::list<
      ::Tags::Mesh<volume_dim>,
      ::Tags::Mortars<::Tags::Mesh<volume_dim - 1>, volume_dim>,
      ::Tags::Mortars<::Tags::MortarSize<volume_dim - 1>, volume_dim>>;

  static void apply(
      const gsl::not_null<db::item_type<dt_variables_tag>*> dt_variables,
      const gsl::not_null<
          db::item_type<::Tags::Mortars<mortar_data_tag, volume_dim>>*>
          all_mortar_data,
      const Mesh<volume_dim>& volume_mesh,
      const db::item_type<::Tags::Mortars<::Tags::Mesh<volume_dim - 1>,
                                          volume_dim>>& mortar_meshes,
      const db::item_type<::Tags::Mortars<::Tags::MortarSize<volume_dim - 1>,
                                          volume_dim>>& mortar_sizes,
      const typename NumericalFluxComputerTag::type&
          normal_dot_numerical_flux_computer) noexcept {
    // Iterate over all mortars
    for (auto& mortar_id_and_data : *all_mortar_data) {
      // Retrieve mortar data
      const auto& mortar_id = mortar_id_and_data.first;
      auto& mortar_data = mortar_id_and_data.second;
      const auto& direction = mortar_id.first;
      const size_t dimension = direction.dimension();
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);
      const auto face_mesh = volume_mesh.slice_away(dimension);
      const size_t extent_perpendicular_to_face =
          volume_mesh.extents(dimension);

      // Extract local and remote data
      const auto& extracted_mortar_data = mortar_data.extract();
      const LocalData& local_data = extracted_mortar_data.first;
      const RemoteData& remote_data = extracted_mortar_data.second;

      // Compute the Tags::NormalDotNumericalFlux from local and remote data
      db::item_type<normal_dot_numerical_fluxes_tag>
          normal_dot_numerical_fluxes{mortar_mesh.number_of_grid_points()};
      MortarHelpers_detail::apply_normal_dot_numerical_flux(
          make_not_null(&normal_dot_numerical_fluxes),
          normal_dot_numerical_flux_computer, local_data.mortar_data,
          remote_data);

      // Subtract the local Tags::NormalDotFlux
      tmpl::for_each<
          db::get_variables_tags_list<normal_dot_numerical_fluxes_tag>>([
        &normal_dot_numerical_fluxes, &local_data
      ](const auto flux_tag) noexcept {
        using FluxTag = tmpl::type_from<decltype(flux_tag)>;
        auto& numerical_flux = get<FluxTag>(normal_dot_numerical_fluxes);
        const auto& local_flux =
            get<::Tags::NormalDotFlux<db::remove_tag_prefix<FluxTag>>>(
                local_data.mortar_data);
        for (size_t i = 0; i < numerical_flux.size(); ++i) {
          numerical_flux[i] -= local_flux[i];
        }
      });

      // Set up variables for the flux contributions that will be added to the
      // volume data
      db::item_type<dt_variables_tag> face_flux;

      // Project from the mortar back to the face if needed
      if (face_mesh != mortar_mesh or
          std::any_of(
              mortar_size.begin(),
              mortar_size.end(), [](const Spectral::MortarSize s) noexcept {
                return s != Spectral::MortarSize::Full;
              })) {
        face_flux = project_from_mortar(std::move(normal_dot_numerical_fluxes),
                                        face_mesh, mortar_mesh, mortar_size);
      } else {
        face_flux = std::move(normal_dot_numerical_fluxes);
      }

      // Lift flux to the volume. We still only need to provide it on the face
      // because it is zero everywhere else.
      face_flux *=
          -0.5 * get(local_data.magnitude_of_face_normal) *
          (extent_perpendicular_to_face * (extent_perpendicular_to_face - 1));

      // Add the flux contribution to the volume data
      add_slice_to_data(dt_variables, face_flux, volume_mesh.extents(),
                        dimension,
                        index_to_slice_at(volume_mesh.extents(), direction));
    }
  }
};  // namespace FluxLifting
}  // namespace FluxLifting
}  // namespace dg
