// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace BoundarySchemes {

template <size_t Dim, typename VariablesTag, typename NumericalFluxComputerTag,
          typename TemporalIdTag, typename TimeStepperTag>
struct StrongFirstOrderLocalTimeStepping {
 private:
  using base = StrongFirstOrder<Dim, VariablesTag, NumericalFluxComputerTag,
                                TemporalIdTag>;

 public:
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

  using BoundaryData = typename base::BoundaryData;
  using boundary_data_computer = typename base::boundary_data_computer;

  using mortar_data_tag =
      Tags::BoundaryHistory<BoundaryData, BoundaryData,
                            db::const_item_type<variables_tag>>;

  using return_tags =
      tmpl::list<variables_tag, ::Tags::Mortars<mortar_data_tag, Dim>>;
  using argument_tags =
      tmpl::list<::Tags::Mesh<Dim>, ::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
                 NumericalFluxComputerTag, TimeStepperTag, Tags::TimeStep>;

  static void apply(
      const gsl::not_null<db::item_type<variables_tag>*> variables,
      const gsl::not_null<db::item_type<::Tags::Mortars<mortar_data_tag, Dim>>*>
          all_mortar_data,
      const Mesh<Dim>& volume_mesh,
      const db::const_item_type<::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>>&
          mortar_meshes,
      const db::const_item_type<
          ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
      const db::const_item_type<NumericalFluxComputerTag>&
          normal_dot_numerical_flux_computer,
      const db::const_item_type<TimeStepperTag>& time_stepper,
      const db::const_item_type<Tags::TimeStep>& time_step) noexcept {
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
      // This lambda must only capture quantities that are
      // independent of the simulation state.
      const auto coupling =
          [
            &face_mesh, &mortar_mesh, &mortar_size,
            &extent_perpendicular_to_face, &normal_dot_numerical_flux_computer
          ](const BoundaryData& local_data,
            const BoundaryData& remote_data) noexcept {
        return StrongFirstOrder_detail::compute_boundary_flux_contribution<
            variables_tag>(
            local_data, remote_data, normal_dot_numerical_flux_computer,
            get<magnitude_of_face_normal_tag>(local_data.extra_data),
            extent_perpendicular_to_face, face_mesh, mortar_mesh, mortar_size);
      };

      const auto lifted_data = time_stepper.compute_boundary_delta(
          coupling, make_not_null(&mortar_data), time_step);

      // Add the flux contribution to the volume data
      add_slice_to_data(variables, lifted_data, volume_mesh.extents(),
                        dimension,
                        index_to_slice_at(volume_mesh.extents(), direction));
    }
  }
};
}  // namespace BoundarySchemes
}  // namespace dg
