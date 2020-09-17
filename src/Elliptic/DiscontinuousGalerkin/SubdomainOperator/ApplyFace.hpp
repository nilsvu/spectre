// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <boost/range/join.hpp>
#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Elliptic/DiscontinuousGalerkin/FirstOrderOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TupleSlice.hpp"

namespace elliptic::dg::subdomain_operator {

namespace detail {

template <bool DontUnmap, typename MapOrValue, typename Key>
const auto& unmap(const MapOrValue& map_or_value, const Key& key) noexcept {
  if constexpr (DontUnmap) {
    return map_or_value;
  } else {
    return map_or_value.at(key);
  }
}

template <typename... MapsOrValues, typename Key, typename... DontUnmap>
auto unmap_all(const std::tuple<MapsOrValues...>& maps_or_values,
               const Key& key, tmpl::list<DontUnmap...>) noexcept {
  return std::apply(
      [&key](const auto&... expanded_maps_or_values) noexcept {
        return std::forward_as_tuple(
            unmap<DontUnmap::value>(expanded_maps_or_values, key)...);
      },
      maps_or_values);
}

}  // namespace detail

/*!
 * \brief Apply the DG subdomain operator to a face of the subdomain's central
 * element.
 *
 * This function is intended to be invoked on all faces of the central element
 * in turn. It visits not only both sides of the mortars that touch the face,
 * but also the neighboring element including its faces to other elements that
 * may or may not overlap with the subdomain. When visiting all these faces it
 * either adds contributions to the `result` right away, or it caches data on
 * one side of a mortar to consume it when it encounters the mortar from the
 * other side. The first and second encounter with a mortar may occur in
 * different invocations of this function, i.e. when calling this function on
 * different faces of the central element. Only once this function has been
 * called on all faces of the central element is it guaranteed that all mortars
 * in the subdomain have been handled.
 *
 * See the `LinearSolver::Schwarz::Schwarz` documentation for details on
 * subdomain operators and for an illustration of the subdomain geometry.
 */
template <
    bool IsExternalBoundary, typename PrimalFields, typename AuxiliaryFields,
    typename SourcesComputerType, bool MassiveOperator, size_t Dim,
    typename FluxesComputerType, typename NumericalFluxesComputerType,
    typename... FluxesArgs, typename... OverlapFluxesArgs,
    typename... OverlapFluxesArgIsFromCenter, typename... OverlapFluxesFaceArgs,
    typename... OverlapFluxesArgIsInVolume, typename... OverlapSourcesArgs,
    typename... OverlapSourcesArgIsFromCenter, typename ResultTags,
    typename ArgTags, typename BoundaryData, typename AllFieldsTags,
    typename FluxesTags, typename DivFluxesTags>
static void apply_face(
    const gsl::not_null<
        LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ResultTags>*>
        result,
    const Direction<Dim>& direction, const Element<Dim>& element,
    const Mesh<Dim>& mesh, const FluxesComputerType& fluxes_computer,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const tnsr::i<DataVector, Dim>& face_normal,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& surface_jacobian,
    const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& mortar_meshes,
    const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& mortar_sizes,
    const LinearSolver::Schwarz::OverlapMap<Dim, size_t>& all_overlap_extents,
    const LinearSolver::Schwarz::OverlapMap<Dim, Mesh<Dim>>& all_overlap_meshes,
    const LinearSolver::Schwarz::OverlapMap<Dim, Element<Dim>>&
        all_overlap_elements,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>>&
        all_overlap_inv_jacobians,
    const LinearSolver::Schwarz::OverlapMap<Dim, Scalar<DataVector>>&
        all_overlap_det_jacobians,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, DirectionMap<Dim, tnsr::i<DataVector, Dim>>>&
        all_overlap_face_normals,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, DirectionMap<Dim, Scalar<DataVector>>>&
        all_overlap_face_normal_magnitudes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, DirectionMap<Dim, Scalar<DataVector>>>&
        all_overlap_surface_jacobians,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, Mesh<Dim - 1>>>& all_overlap_mortar_meshes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>&
        all_overlap_mortar_sizes,
    const LinearSolver::Schwarz::OverlapMap<Dim,
                                            ::dg::MortarMap<Dim, Mesh<Dim>>>
        all_overlap_neighbor_meshes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, Scalar<DataVector>>>
        all_overlap_neighbor_face_normal_magnitudes,
    const LinearSolver::Schwarz::OverlapMap<Dim,
                                            ::dg::MortarMap<Dim, Mesh<Dim - 1>>>
        all_overlap_neighbor_mortar_meshes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>
        all_overlap_neighbor_mortar_sizes,
    const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<OverlapFluxesArgs...>& all_overlap_fluxes_args,
    const tmpl::list<OverlapFluxesArgIsFromCenter...> /*meta*/,
    const std::tuple<OverlapFluxesFaceArgs...>& all_overlap_fluxes_face_args,
    const tmpl::list<OverlapFluxesArgIsInVolume...> /*meta*/,
    const std::tuple<OverlapSourcesArgs...>& all_overlap_sources_args,
    const tmpl::list<OverlapSourcesArgIsFromCenter...> /*meta*/,
    const LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ArgTags>&
        operand,
    const Variables<FluxesTags>& central_fluxes,
    const Variables<DivFluxesTags>& central_div_fluxes,
    const gsl::not_null<std::unordered_map<
        std::pair<ElementId<Dim>, ::dg::MortarId<Dim>>, BoundaryData,
        boost::hash<std::pair<ElementId<Dim>, ::dg::MortarId<Dim>>>>*>
        neighbors_boundary_data,
    const gsl::not_null<Variables<FluxesTags>*> central_fluxes_on_face,
    const gsl::not_null<Variables<DivFluxesTags>*> central_div_fluxes_on_face,
    const gsl::not_null<FixedHashMap<maximum_number_of_neighbors(Dim),
                                     ElementId<Dim>, Variables<AllFieldsTags>>*>
        buffered_neighbor_extended_vars,
    const gsl::not_null<FixedHashMap<maximum_number_of_neighbors(Dim),
                                     ElementId<Dim>, Variables<FluxesTags>>*>
        buffered_neighbor_fluxes,
    const gsl::not_null<FixedHashMap<maximum_number_of_neighbors(Dim),
                                     ElementId<Dim>, Variables<DivFluxesTags>>*>
        buffered_neighbor_div_fluxes,
    const gsl::not_null<
        FixedHashMap<maximum_number_of_neighbors(Dim), ElementId<Dim>,
                     DirectionMap<Dim, Variables<FluxesTags>>>*>
        buffered_neighbor_fluxes_on_face,
    const gsl::not_null<
        FixedHashMap<maximum_number_of_neighbors(Dim), ElementId<Dim>,
                     DirectionMap<Dim, Variables<DivFluxesTags>>>*>
        buffered_neighbor_div_fluxes_on_face,
    const gsl::not_null<FixedHashMap<maximum_number_of_neighbors(Dim),
                                     ElementId<Dim>, Variables<AllFieldsTags>>*>
        buffered_neighbor_result_extended) noexcept {
  const size_t dimension = direction.dimension();
  const auto face_mesh = mesh.slice_away(dimension);
  const size_t slice_index = index_to_slice_at(mesh.extents(), direction);

  // Slice volume fluxes and their divergence to the face
  data_on_slice(central_fluxes_on_face, central_fluxes, mesh.extents(),
                dimension, slice_index);
  data_on_slice(central_div_fluxes_on_face, central_div_fluxes, mesh.extents(),
                dimension, slice_index);

  // Compute the normal dot fluxes
  const auto normal_dot_central_fluxes =
      normal_dot_flux<AllFieldsTags>(face_normal, *central_fluxes_on_face);

  // Assemble boundary data
  const auto center_boundary_data_on_face = elliptic::dg::package_boundary_data(
      numerical_fluxes_computer, fluxes_computer, mesh, direction, face_mesh,
      face_normal, magnitude_of_face_normal, normal_dot_central_fluxes,
      *central_div_fluxes_on_face, fluxes_args, AuxiliaryFields{});

  // Decide whether we're on an external boundary where we need to impose
  // boundary conditions instead of looking at the other side of the mortar for
  // data
  if constexpr (IsExternalBoundary) {
    const auto central_vars_on_face = Variables<AllFieldsTags>(data_on_slice(
        operand.element_data, mesh.extents(), dimension, slice_index));
    BoundaryData remote_boundary_data{};
    elliptic::dg::package_exterior_boundary_data<PrimalFields, AuxiliaryFields>(
        make_not_null(&remote_boundary_data), central_vars_on_face,
        *central_div_fluxes_on_face, mesh, direction, face_mesh, face_normal,
        magnitude_of_face_normal, fluxes_computer, numerical_fluxes_computer,
        fluxes_args);
    // No projections necessary since an external boundary mortar covers the
    // full face
    elliptic::dg::apply_boundary_contribution<MassiveOperator>(
        make_not_null(&result->element_data), numerical_fluxes_computer,
        center_boundary_data_on_face, std::move(remote_boundary_data),
        magnitude_of_face_normal, surface_jacobian, mesh, direction, face_mesh,
        make_array<Dim - 1>(Spectral::MortarSize::Full));
  } else {
    const auto& neighbors = element.neighbors().at(direction);
    const auto& orientation = neighbors.orientation();
    const auto overlap_direction_in_neighbor =
        orientation(direction.opposite());
    const size_t overlap_dimension_in_neighbor =
        overlap_direction_in_neighbor.dimension();

    // Iterate over all neighbors in this direction, computing their volume
    // operator and all of their boundary contributions within the subdomain.
    // Note that data on overlaps is oriented according to the neighbor that it
    // is on, as is all geometric information that we have on the neighbor. Only
    // when data cross element boundaries do we need to re-orient.
    for (const auto& neighbor_id : neighbors) {
      const auto overlap_id = std::make_pair(direction, neighbor_id);
      const size_t overlap_extent = all_overlap_extents.at(overlap_id);
      const auto& neighbor_mesh = all_overlap_meshes.at(overlap_id);
      // The mortar ID is technically the same as the overlap ID (just a pair),
      // but it refers to the mortar, not the overlap region, so we name it
      // accordingly
      const auto& mortar_id = overlap_id;
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);

      // Project the central element's boundary data to this mortar
      const auto center_boundary_data =
          ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? center_boundary_data_on_face.project_to_mortar(
                    face_mesh, mortar_mesh, mortar_size)
              : center_boundary_data_on_face;

      // Intercept empty overlaps. In the unlikely case that overlaps have zero
      // extent (which is fairly useless, except for testing) the subdomain is
      // identical to the central element and no communication with neighbors is
      // necessary. We can just handle the mortar between central element and
      // neighbor and continue.
      if (UNLIKELY(overlap_extent == 0)) {
        // Make sure the result has no overlap data in this direction because we
        // won't mutate it
        result->overlap_data.erase(overlap_id);
        // Assume the data on the other side of the mortar is zero and apply the
        // contribution to the central element
        BoundaryData remote_boundary_data{};
        const auto mortar_id_from_neighbor =
            std::make_pair(overlap_direction_in_neighbor, element.id());
        numerical_fluxes_computer.package_zero_data(
            make_not_null(&remote_boundary_data), neighbor_mesh,
            overlap_direction_in_neighbor,
            all_overlap_face_normal_magnitudes.at(overlap_id)
                .at(overlap_direction_in_neighbor));
        const auto neighbor_face_mesh =
            neighbor_mesh.slice_away(overlap_dimension_in_neighbor);
        const auto& neighbor_mortar_mesh =
            all_overlap_mortar_meshes.at(overlap_id)
                .at(mortar_id_from_neighbor);
        ASSERT(mortar_mesh == neighbor_mortar_mesh,
               "Mortar meshes should be the same from both sides");
        const auto& neighbor_mortar_size =
            all_overlap_mortar_sizes.at(overlap_id).at(mortar_id_from_neighbor);
        if (::dg::needs_projection(neighbor_face_mesh, neighbor_mortar_mesh,
                                   neighbor_mortar_size)) {
          remote_boundary_data = remote_boundary_data.project_to_mortar(
              neighbor_face_mesh, neighbor_mortar_mesh, neighbor_mortar_size);
        }
        if (not orientation.is_aligned()) {
          remote_boundary_data.orient_on_slice(neighbor_mortar_mesh.extents(),
                                               overlap_dimension_in_neighbor,
                                               orientation.inverse_map());
        }
        elliptic::dg::apply_boundary_contribution<MassiveOperator>(
            make_not_null(&result->element_data), numerical_fluxes_computer,
            center_boundary_data, remote_boundary_data,
            magnitude_of_face_normal, surface_jacobian, mesh, direction, mortar_mesh,
            mortar_size);
        continue;
      }

      const auto& overlap_data = operand.overlap_data.at(overlap_id);
      const auto& neighbor_inv_jacobian =
          all_overlap_inv_jacobians.at(overlap_id);
      const auto& neighbor_det_jacobian =
          all_overlap_det_jacobians.at(overlap_id);

      // Extend the overlap data to the full neighbor mesh by padding it with
      // zeros. This is necessary because spectral operators such as derivatives
      // require data on the full mesh.
      LinearSolver::Schwarz::extended_overlap_data(
          make_not_null(&(*buffered_neighbor_extended_vars)[neighbor_id]),
          overlap_data, neighbor_mesh.extents(), overlap_extent,
          overlap_direction_in_neighbor);
      const auto& neighbor_data =
          buffered_neighbor_extended_vars->at(neighbor_id);
      auto& neighbor_fluxes = (*buffered_neighbor_fluxes)[neighbor_id];
      auto& neighbor_div_fluxes = (*buffered_neighbor_div_fluxes)[neighbor_id];
      auto& neighbor_result_extended =
          (*buffered_neighbor_result_extended)[neighbor_id];

      // Compute the volume contribution in the neighbor from the extended
      // overlap data
      elliptic::first_order_operator<PrimalFields, AuxiliaryFields,
                                     SourcesComputerType>(
          make_not_null(&neighbor_result_extended),
          make_not_null(&neighbor_fluxes), make_not_null(&neighbor_div_fluxes),
          neighbor_data, neighbor_mesh, neighbor_inv_jacobian, fluxes_computer,
          detail::unmap_all(all_overlap_fluxes_args, overlap_id,
                            tmpl::list<OverlapFluxesArgIsFromCenter...>{}),
          detail::unmap_all(all_overlap_sources_args, overlap_id,
                            tmpl::list<OverlapSourcesArgIsFromCenter...>{}));
       if constexpr (MassiveOperator) {
         apply_mass(make_not_null(&neighbor_result_extended), neighbor_mesh,
                    neighbor_det_jacobian);
       }

      // Iterate over the neighbor's mortars to compute boundary data. For the
      // mortars to the subdomain center, to external boundaries and to elements
      // that are not part of the subdomain we can handle the boundary
      // contribution to the neighbor and to the potential other element right
      // away. For mortars to other neighbors within the subdomain we cache the
      // boundary data and handle the boundary contribution in the second pass
      // over the same mortar.
      const auto& neighbor = all_overlap_elements.at(overlap_id);
      for (const auto& neighbor_face_direction :
           boost::join(neighbor.internal_boundaries(),
                       neighbor.external_boundaries())) {
        // We can skip neighbor mortars that face away from the subdomain center
        // because they only contribute to points on the face, which are never
        // part of the subdomain. WARNING: This may not hold anymore when
        // changes to the DG operator are made, e.g. how it applies the mass
        // matrix. See `LinearSolver::Schwarz::overlap_extents` for details.
        // When that happens this face may need to be handled like the others.
        if (neighbor_face_direction ==
            overlap_direction_in_neighbor.opposite()) {
          ASSERT(neighbor_mesh.quadrature(overlap_dimension_in_neighbor) ==
                     Spectral::Quadrature::GaussLobatto,
                 "Assuming Gauss-Lobatto grid");
          continue;
        }
        const size_t neighbor_face_dimension =
            neighbor_face_direction.dimension();
        const size_t neighbor_face_slice_index =
            index_to_slice_at(neighbor_mesh.extents(), neighbor_face_direction);
        const auto neighbor_face_mesh =
            neighbor_mesh.slice_away(neighbor_face_dimension);
        const auto& neighbor_face_normal =
            all_overlap_face_normals.at(overlap_id).at(neighbor_face_direction);
        const auto& neighbor_face_normal_magnitude =
            all_overlap_face_normal_magnitudes.at(overlap_id)
                .at(neighbor_face_direction);
        const auto& neighbor_surface_jacobian =
            all_overlap_surface_jacobians.at(overlap_id)
                .at(neighbor_face_direction);

        // Compute the boundary data on the neighbor's local side of the
        // mortar
        auto& neighbor_fluxes_on_face =
            (*buffered_neighbor_fluxes_on_face)[neighbor_id]
                                               [neighbor_face_direction];
        data_on_slice(make_not_null(&neighbor_fluxes_on_face), neighbor_fluxes,
                      neighbor_mesh.extents(), neighbor_face_dimension,
                      neighbor_face_slice_index);
        auto& neighbor_div_fluxes_on_face =
            (*buffered_neighbor_div_fluxes_on_face)[neighbor_id]
                                                   [neighbor_face_direction];
        data_on_slice(make_not_null(&neighbor_div_fluxes_on_face),
                      neighbor_div_fluxes, neighbor_mesh.extents(),
                      neighbor_face_dimension, neighbor_face_slice_index);
        const auto neighbor_normal_dot_fluxes = normal_dot_flux<AllFieldsTags>(
            neighbor_face_normal, neighbor_fluxes_on_face);
        const auto neighbor_fluxes_args_on_face = detail::unmap_all(
            detail::unmap_all(all_overlap_fluxes_face_args, overlap_id,
                              tmpl::list<OverlapFluxesArgIsFromCenter...>{}),
            neighbor_face_direction,
            tmpl::list<OverlapFluxesArgIsInVolume...>{});
        auto neighbor_local_boundary_data = elliptic::dg::package_boundary_data(
            numerical_fluxes_computer, fluxes_computer, neighbor_mesh,
            neighbor_face_direction, neighbor_face_mesh, neighbor_face_normal,
            neighbor_face_normal_magnitude, neighbor_normal_dot_fluxes,
            neighbor_div_fluxes_on_face, neighbor_fluxes_args_on_face,
            AuxiliaryFields{});

        if (neighbor.external_boundaries().count(neighbor_face_direction) ==
            1) {
          // This is an external boundary of the neighbor. We apply the
          // boundary contribution directly to the neighbor.
          const auto neighbor_face_data =
              data_on_slice(neighbor_data, neighbor_mesh.extents(),
                            neighbor_face_dimension, neighbor_face_slice_index);
          BoundaryData neighbor_remote_boundary_data{};
          elliptic::dg::package_exterior_boundary_data<PrimalFields,
                                                       AuxiliaryFields>(
              make_not_null(&neighbor_remote_boundary_data), neighbor_face_data,
              neighbor_div_fluxes_on_face, neighbor_mesh,
              neighbor_face_direction, neighbor_face_mesh, neighbor_face_normal,
              neighbor_face_normal_magnitude, fluxes_computer,
              numerical_fluxes_computer, neighbor_fluxes_args_on_face);
          elliptic::dg::apply_boundary_contribution<MassiveOperator>(
              make_not_null(&neighbor_result_extended),
              numerical_fluxes_computer, neighbor_local_boundary_data,
              neighbor_remote_boundary_data, neighbor_face_normal_magnitude,
              neighbor_surface_jacobian, neighbor_mesh, neighbor_face_direction,
              neighbor_face_mesh,
              make_array<Dim - 1>(Spectral::MortarSize::Full));
        } else {
          for (const auto& neighbors_neighbor_id :
               neighbor.neighbors().at(neighbor_face_direction)) {
            const auto neighbor_mortar_id =
                std::pair(neighbor_face_direction, neighbors_neighbor_id);
            const auto& neighbor_mortar_mesh =
                all_overlap_mortar_meshes.at(overlap_id).at(neighbor_mortar_id);
            const auto& neighbor_mortar_size =
                all_overlap_mortar_sizes.at(overlap_id).at(neighbor_mortar_id);

            // Project to the neighbor's local side of the mortar
            auto projected_neighbor_local_boundary_data =
                ::dg::needs_projection(neighbor_face_mesh, neighbor_mortar_mesh,
                                       neighbor_mortar_size)
                    ? neighbor_local_boundary_data.project_to_mortar(
                          neighbor_face_mesh, neighbor_mortar_mesh,
                          neighbor_mortar_size)
                    : neighbor_local_boundary_data;

            // Decide what to do based on what's on the other side of the mortar
            BoundaryData neighbor_remote_boundary_data{};
            if (neighbors_neighbor_id == element.id()) {
              // This is the mortar to the subdomain center. We apply the
              // boundary contribution both to the subdomain center and to the
              // neighbor. First, apply the boundary contribution to the central
              // element
              ASSERT(mortar_mesh == neighbor_mortar_mesh,
                     "Mortar meshes should be the same from both sides");
              auto reoriented_neighbor_boundary_data =
                  projected_neighbor_local_boundary_data;
              if (not orientation.is_aligned()) {
                reoriented_neighbor_boundary_data.orient_on_slice(
                    neighbor_mortar_mesh.extents(),
                    overlap_dimension_in_neighbor, orientation.inverse_map());
              }
              elliptic::dg::apply_boundary_contribution<MassiveOperator>(
                  make_not_null(&result->element_data),
                  numerical_fluxes_computer, center_boundary_data,
                  std::move(reoriented_neighbor_boundary_data),
                  magnitude_of_face_normal, surface_jacobian, mesh, direction,
                  mortar_mesh, mortar_size);
              // Second, prepare applying the boundary contribution to the
              // neighbor
              neighbor_remote_boundary_data = center_boundary_data;
              if (not orientation.is_aligned()) {
                neighbor_remote_boundary_data.orient_on_slice(
                    mortar_mesh.extents(), dimension, orientation);
              }
            } else {
              // This is an internal boundary to another element, which may or
              // may not overlap with the subdomain.
              const auto& neighbors_neighbor_orientation =
                  all_overlap_elements.at(overlap_id)
                      .neighbors()
                      .at(neighbor_face_direction)
                      .orientation();
              const auto direction_from_neighbors_neighbor =
                  neighbors_neighbor_orientation(
                      neighbor_face_direction.opposite());
              const auto mortar_id_from_neighbors_neighbor = std::make_pair(
                  direction_from_neighbors_neighbor, neighbor_id);
              // Determine whether the neighbor's neighbor overlaps with the
              // subdomain and find its overlap ID if it does.
              const auto neighbors_neighbor_overlap_id =
                  [&all_overlap_mortar_meshes, &neighbors_neighbor_id,
                   &mortar_id_from_neighbors_neighbor]() noexcept
                  -> std::optional<LinearSolver::Schwarz::OverlapId<Dim>> {
                for (const auto& overlap_id_and_mortar_meshes :
                     all_overlap_mortar_meshes) {
                  const auto& local_overlap_id =
                      overlap_id_and_mortar_meshes.first;
                  if (local_overlap_id.second != neighbors_neighbor_id) {
                    continue;
                  }
                  const auto& local_mortar_meshes =
                      overlap_id_and_mortar_meshes.second;
                  for (const auto& local_mortar_id_and_mesh :
                       local_mortar_meshes) {
                    const auto& local_mortar_id =
                        local_mortar_id_and_mesh.first;
                    if (local_mortar_id == mortar_id_from_neighbors_neighbor) {
                      return local_overlap_id;
                    }
                  }
                }
                return std::nullopt;
              }();
              if (neighbors_neighbor_overlap_id) {
                // The neighbor's neighbor overlaps with the subdomain, so we
                // can retrieve data from it. We store the data on one side of
                // the mortar in a cache, so when encountering this mortar the
                // second time (from its other side) we apply the boundary
                // contributions to both sides.
                const auto found_neighbors_neighbor_boundary_data =
                    neighbors_boundary_data->find(
                        std::make_pair(neighbors_neighbor_id,
                                       mortar_id_from_neighbors_neighbor));
                if (found_neighbors_neighbor_boundary_data ==
                    neighbors_boundary_data->end()) {
                  (*neighbors_boundary_data)[std::make_pair(
                      neighbor_id, neighbor_mortar_id)] =
                      projected_neighbor_local_boundary_data;

                  // Skip applying the boundary contribution from this mortar
                  // because we don't have the remote data available yet. We
                  // apply the boundary contribution when re-visiting this
                  // mortar from the other side.
                  continue;

                } else {
                  neighbor_remote_boundary_data = std::move(
                      neighbors_boundary_data
                          ->extract(found_neighbors_neighbor_boundary_data)
                          .mapped());

                  // Apply the boundary contribution also to the other side of
                  // the mortar that we had previously skipped
                  auto reoriented_neighbor_local_boundary_data =
                      projected_neighbor_local_boundary_data;
                  if (not neighbors_neighbor_orientation.is_aligned()) {
                    reoriented_neighbor_local_boundary_data.orient_on_slice(
                        neighbor_mortar_mesh.extents(),
                        neighbor_face_direction.dimension(),
                        neighbors_neighbor_orientation);
                  }
                  const auto& neighbors_neighbor_mesh =
                      all_overlap_meshes.at(*neighbors_neighbor_overlap_id);
                  const auto& neighbors_neighbor_overlap_extents =
                      all_overlap_extents.at(*neighbors_neighbor_overlap_id);
                  const auto overlap_direction_from_neighbors_neighbor =
                      element.neighbors()
                          .at(neighbors_neighbor_overlap_id->first)
                          .orientation()(
                              neighbors_neighbor_overlap_id->first.opposite());
                  const auto& neighbors_neighbor_mortar_mesh =
                      all_overlap_mortar_meshes
                          .at(*neighbors_neighbor_overlap_id)
                          .at(mortar_id_from_neighbors_neighbor);
                  ASSERT(neighbors_neighbor_mortar_mesh == neighbor_mortar_mesh,
                         "Mortar meshes should be the same from both sides");
                  const auto& neighbors_neighbor_mortar_size =
                      all_overlap_mortar_sizes
                          .at(*neighbors_neighbor_overlap_id)
                          .at(mortar_id_from_neighbors_neighbor);
                  // It would be nice if we could apply directly to the
                  // overlap-restricted data instead of
                  // extending-then-restricting. At least we can re-use the
                  // other neighbor's buffer, so no memory needs to be
                  // re-allocated for this operation.
                  auto& neighbors_neighbor_result_extended =
                      (*buffered_neighbor_result_extended)
                          [neighbors_neighbor_id];
                  LinearSolver::Schwarz::extended_overlap_data(
                      make_not_null(&neighbors_neighbor_result_extended),
                      result->overlap_data.at(*neighbors_neighbor_overlap_id),
                      neighbors_neighbor_mesh.extents(),
                      neighbors_neighbor_overlap_extents,
                      overlap_direction_from_neighbors_neighbor);
                  elliptic::dg::apply_boundary_contribution<MassiveOperator>(
                      make_not_null(&neighbors_neighbor_result_extended),
                      numerical_fluxes_computer, neighbor_remote_boundary_data,
                      std::move(reoriented_neighbor_local_boundary_data),
                      all_overlap_face_normal_magnitudes
                          .at(*neighbors_neighbor_overlap_id)
                          .at(direction_from_neighbors_neighbor),
                      all_overlap_surface_jacobians
                          .at(*neighbors_neighbor_overlap_id)
                          .at(direction_from_neighbors_neighbor),
                      neighbors_neighbor_mesh,
                      direction_from_neighbors_neighbor,
                      neighbors_neighbor_mortar_mesh,
                      neighbors_neighbor_mortar_size);
                  LinearSolver::Schwarz::data_on_overlap(
                      make_not_null(&result->overlap_data.at(
                          *neighbors_neighbor_overlap_id)),
                      neighbors_neighbor_result_extended,
                      neighbors_neighbor_mesh.extents(),
                      neighbors_neighbor_overlap_extents,
                      overlap_direction_from_neighbors_neighbor);

                  // Prepare applying the boundary contribution to the neighbor
                  if (not neighbors_neighbor_orientation.is_aligned()) {
                    neighbor_remote_boundary_data.orient_on_slice(
                        neighbors_neighbor_mortar_mesh.extents(),
                        direction_from_neighbors_neighbor.dimension(),
                        neighbors_neighbor_orientation.inverse_map());
                  }
                }
              } else {
                // The neighbor's neighbor does not overlap with the subdomain,
                // so we assume the data on it is zero. We have to do
                // projections and orientations even though the data is zero to
                // handle the element size correctly for computing penalties.
                const auto& neighbors_neighbor_mesh =
                    all_overlap_neighbor_meshes.at(overlap_id)
                        .at(neighbor_mortar_id);
                numerical_fluxes_computer.package_zero_data(
                    make_not_null(&neighbor_remote_boundary_data),
                    neighbors_neighbor_mesh, direction_from_neighbors_neighbor,
                    all_overlap_neighbor_face_normal_magnitudes.at(overlap_id)
                        .at(neighbor_mortar_id));
                const auto neighbors_neighbor_face_mesh =
                    neighbors_neighbor_mesh.slice_away(
                        direction_from_neighbors_neighbor.dimension());
                const auto& neighbors_neighbor_mortar_mesh =
                    all_overlap_neighbor_mortar_meshes.at(overlap_id)
                        .at(neighbor_mortar_id);
                const auto& neighbors_neighbor_mortar_size =
                    all_overlap_neighbor_mortar_sizes.at(overlap_id)
                        .at(neighbor_mortar_id);
                if (::dg::needs_projection(neighbors_neighbor_face_mesh,
                                           neighbors_neighbor_mortar_mesh,
                                           neighbors_neighbor_mortar_size)) {
                  neighbor_remote_boundary_data =
                      neighbor_remote_boundary_data.project_to_mortar(
                          neighbors_neighbor_face_mesh,
                          neighbors_neighbor_mortar_mesh,
                          neighbors_neighbor_mortar_size);
                }
                if (not neighbors_neighbor_orientation.is_aligned()) {
                  neighbor_remote_boundary_data.orient_on_slice(
                      neighbors_neighbor_mortar_mesh.extents(),
                      direction_from_neighbors_neighbor.dimension(),
                      neighbors_neighbor_orientation.inverse_map());
                }
              }
            }
            elliptic::dg::apply_boundary_contribution<MassiveOperator>(
                make_not_null(&neighbor_result_extended),
                numerical_fluxes_computer,
                projected_neighbor_local_boundary_data,
                neighbor_remote_boundary_data, neighbor_face_normal_magnitude,
                neighbor_surface_jacobian, neighbor_mesh,
                neighbor_face_direction, neighbor_mortar_mesh,
                neighbor_mortar_size);
          }  // neighbor mortars
        }    // if external neighbor face
      }      // neighbor faces
      LinearSolver::Schwarz::data_on_overlap(
          make_not_null(&result->overlap_data.at(overlap_id)),
          neighbor_result_extended, neighbor_mesh.extents(), overlap_extent,
          overlap_direction_in_neighbor);
    }  // neighbors
  }    // if constexpr (IsExternalBoundary)
}

}  // namespace elliptic::dg::subdomain_operator
