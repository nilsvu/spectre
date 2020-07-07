// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"

/// \cond
template <size_t Dim>
struct DomainCreator;
/// \endcond

/// Functionality for testing elliptic DG operators
namespace TestHelpers::elliptic::dg {

/// Prefix tag that represents the elliptic DG operator applied to fields.
template <typename Tag>
struct OperatorAppliedTo : db::PrefixTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};

/// An element in a DG domain
template <size_t Dim>
struct DgElement {
  Mesh<Dim> mesh;
  Element<Dim> element;
  ElementMap<Dim, Frame::Inertial> element_map;
};

/// A set of `DgElement`s that make up the domain
template <size_t Dim>
using ElementArray = std::unordered_map<ElementId<Dim>, DgElement<Dim>>;

/// Construct a `ElementArray` from the `domain_creator
template <size_t Dim>
ElementArray<Dim> create_elements(
    const DomainCreator<Dim>& domain_creator) noexcept;

/// Construct all mortars for the given `element_id`
template <size_t VolumeDim>
::dg::MortarMap<VolumeDim,
                std::pair<Mesh<VolumeDim - 1>, ::dg::MortarSize<VolumeDim - 1>>>
create_mortars(const ElementId<VolumeDim>& element_id,
               const ElementArray<VolumeDim>& dg_elements) noexcept;

namespace detail {
// Dummy tag to check the system's magnitude tag
struct TestTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace detail

/*!
 * \brief Apply the first-order elliptic DG operator to a set of system
 * variables
 *
 * Supply functions that return a `std::tuple` of the system's fluxes and
 * sources arguments to `package_fluxes_args` and `package_sources_args`,
 * respectively.
 *
 * The DG boundary scheme is defined by the functions `package_boundary_data`
 * and `apply_boundary_contribution`. Here's an example how a strong first-order
 * Poisson boundary scheme with a particular numerical flux is defined:
 *
 * \snippet Helpers/Elliptic/Systems/Poisson/DgSchemes.cpp boundary_scheme
 */
template <typename System, typename TagsList, typename PackageFluxesArgs,
          typename PackageSourcesArgs, typename PackageBoundaryData,
          typename ApplyBoundaryContribution, size_t Dim = System::volume_dim,
          typename PrimalFields = typename System::primal_fields,
          typename AuxiliaryFields = typename System::auxiliary_fields,
          typename FluxesComputer = typename System::fluxes,
          typename SourcesComputer = typename System::sources>
Variables<db::wrap_tags_in<OperatorAppliedTo, TagsList>>
apply_first_order_dg_operator(
    const ElementId<Dim>& element_id, const ElementArray<Dim>& dg_elements,
    const std::unordered_map<ElementId<Dim>, Variables<TagsList>>&
        all_variables,
    const FluxesComputer& fluxes_computer,
    PackageFluxesArgs&& package_fluxes_args,
    PackageSourcesArgs&& package_sources_args,
    PackageBoundaryData&& package_boundary_data,
    ApplyBoundaryContribution&& apply_boundary_contribution) {
  static constexpr size_t volume_dim = Dim;
  using Vars = Variables<TagsList>;
  using ResultVars = Variables<db::wrap_tags_in<OperatorAppliedTo, TagsList>>;

  const auto& dg_element = dg_elements.at(element_id);
  const auto& vars = all_variables.at(element_id);

  const size_t num_points = dg_element.mesh.number_of_grid_points();
  ResultVars result{num_points};

  // Compute bulk contribution in central element
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, TagsList,
                                       tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  Variables<fluxes_tags> fluxes{num_points};
  Variables<div_fluxes_tags> div_fluxes{num_points};
  const auto logical_coords = logical_coordinates(dg_element.mesh);
  const auto inv_jacobian = dg_element.element_map.inv_jacobian(logical_coords);
  ::elliptic::first_order_operator<PrimalFields, AuxiliaryFields,
                                   SourcesComputer>(
      make_not_null(&result), make_not_null(&fluxes),
      make_not_null(&div_fluxes), vars, dg_element.mesh, inv_jacobian,
      fluxes_computer, package_fluxes_args(element_id, dg_element),
      package_sources_args(element_id, dg_element));

  // Setup mortars
  const auto mortars = create_mortars(element_id, dg_elements);

  // Add boundary contributions
  for (const auto& mortar : mortars) {
    const auto& mortar_id = mortar.first;
    const auto& mortar_mesh = mortar.second.first;
    const auto& mortar_size = mortar.second.second;
    const auto& direction = mortar_id.first;
    const auto& neighbor_id = mortar_id.second;

    const size_t dimension = direction.dimension();
    const auto face_mesh = dg_element.mesh.slice_away(dimension);
    const size_t face_num_points = face_mesh.number_of_grid_points();
    const size_t slice_index =
        index_to_slice_at(dg_element.mesh.extents(), direction);

    // Compute normalized face normal and magnitude
    auto face_normal =
        unnormalized_face_normal(face_mesh, dg_element.element_map, direction);
    // Assuming Euclidean magnitude for now. Could retrieve the magnitude
    // compute tag from the `system` but then we need to handle its arguments.
    // Or we could pass a magnitude function (pointer) into `apply_dg_operator`,
    // but then we should check it's consistent with the system.
    static_assert(
        std::is_same_v<typename System::template magnitude_tag<detail::TestTag>,
                       ::Tags::EuclideanMagnitude<detail::TestTag>>,
        "Only Euclidean magnitudes are currently supported.");
    const auto magnitude_of_face_normal = magnitude(face_normal);
    for (size_t d = 0; d < volume_dim; d++) {
      face_normal.get(d) /= get(magnitude_of_face_normal);
    }

    // Compute normal dot fluxes
    const auto fluxes_on_face = data_on_slice(fluxes, dg_element.mesh.extents(),
                                              dimension, slice_index);
    const auto normal_dot_fluxes =
        normal_dot_flux<TagsList>(face_normal, fluxes_on_face);

    // Slice flux divergences to face
    const auto div_fluxes_on_face = data_on_slice(
        div_fluxes, dg_element.mesh.extents(), dimension, slice_index);

    // Assemble local boundary data
    const auto fluxes_args_on_face =
        package_fluxes_args(element_id, dg_element, direction);
    auto local_boundary_data = package_boundary_data(
        dg_element.mesh, direction, face_normal, magnitude_of_face_normal,
        normal_dot_fluxes, div_fluxes_on_face, fluxes_args_on_face);
    if (::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
      local_boundary_data = local_boundary_data.project_to_mortar(
          face_mesh, mortar_mesh, mortar_size);
    }

    // Assemble remote boundary data
    std::decay_t<decltype(local_boundary_data)> remote_boundary_data;
    if (neighbor_id == ElementId<volume_dim>::external_boundary_id()) {
      // On exterior ("ghost") faces, manufacture boundary data that represent
      // homogeneous Dirichlet boundary conditions
      const auto vars_on_face = data_on_slice(vars, dg_element.mesh.extents(),
                                              dimension, slice_index);
      Vars ghost_vars{face_num_points};
      ::elliptic::dg::homogeneous_dirichlet_boundary_conditions<PrimalFields>(
          make_not_null(&ghost_vars), vars_on_face);
      const auto ghost_fluxes = std::apply(
          [&ghost_vars, &fluxes_computer](const auto&... fluxes_args) {
            return ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                                  AuxiliaryFields>(
                ghost_vars, fluxes_computer, fluxes_args...);
          },
          fluxes_args_on_face);
      auto remote_face_normal = face_normal;
      for (size_t d = 0; d < volume_dim; d++) {
        remote_face_normal.get(d) *= -1.;
      }
      const auto ghost_normal_dot_fluxes =
          normal_dot_flux<TagsList>(remote_face_normal, ghost_fluxes);
      remote_boundary_data = package_boundary_data(
          dg_element.mesh, direction.opposite(), remote_face_normal,
          magnitude_of_face_normal, ghost_normal_dot_fluxes,
          // Using the div_fluxes from the interior here is fine for Dirichlet
          // boundaries
          div_fluxes_on_face, fluxes_args_on_face);
    } else {
      // On internal boundaries, get neighbor data from all_variables
      const auto& neighbor_orientation =
          dg_element.element.neighbors().at(direction).orientation();
      const auto direction_from_neighbor =
          neighbor_orientation(direction.opposite());
      const auto& neighbor = dg_elements.at(neighbor_id);
      const auto neighbor_face_mesh =
          neighbor.mesh.slice_away(direction_from_neighbor.dimension());
      const auto neighbor_mortars = create_mortars(neighbor_id, dg_elements);
      const auto& neighbor_mortar = neighbor_mortars.at(
          std::make_pair(direction_from_neighbor, element_id));
      const auto& neighbor_mortar_mesh = neighbor_mortar.first;
      const auto& neighbor_mortar_size = neighbor_mortar.second;
      ASSERT(neighbor_mortar_mesh == mortar_mesh,
             "Mismatch between neighboring mortar meshes");
      const auto& remote_vars = all_variables.at(neighbor_id);
      const auto fluxes_args_on_remote_face =
          package_fluxes_args(neighbor_id, neighbor, direction_from_neighbor);
      const auto remote_fluxes = std::apply(
          [&remote_vars, &fluxes_computer](const auto&... fluxes_args) {
            return ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                                  AuxiliaryFields>(
                remote_vars, fluxes_computer, fluxes_args...);
          },
          package_fluxes_args(neighbor_id, neighbor, direction_from_neighbor,
                              fluxes_args_on_remote_face));
      const auto neighbor_logical_coords = logical_coordinates(neighbor.mesh);
      const auto neighbor_inv_jacobian =
          neighbor.element_map.inv_jacobian(logical_coords);
      const auto remote_div_fluxes_on_face = data_on_slice(
          divergence(remote_fluxes, neighbor.mesh, neighbor_inv_jacobian),
          neighbor.mesh.extents(), direction_from_neighbor.dimension(),
          index_to_slice_at(neighbor.mesh.extents(), direction_from_neighbor));
      const auto remote_fluxes_on_face = data_on_slice(
          remote_fluxes, neighbor.mesh.extents(),
          direction_from_neighbor.dimension(),
          index_to_slice_at(neighbor.mesh.extents(), direction_from_neighbor));
      auto remote_face_normal = unnormalized_face_normal(
          neighbor_face_mesh, neighbor.element_map, direction_from_neighbor);
      const auto remote_face_normal_magnitude = magnitude(remote_face_normal);
      for (size_t d = 0; d < volume_dim; d++) {
        remote_face_normal.get(d) /= get(remote_face_normal_magnitude);
      }
      auto remote_normal_dot_fluxes =
          normal_dot_flux<TagsList>(remote_face_normal, remote_fluxes_on_face);
      remote_boundary_data = package_boundary_data(
          neighbor.mesh, direction_from_neighbor, remote_face_normal,
          remote_face_normal_magnitude, remote_normal_dot_fluxes,
          remote_div_fluxes_on_face, fluxes_args_on_remote_face);
      if (::dg::needs_projection(neighbor_face_mesh, neighbor_mortar_mesh,
                                 neighbor_mortar_size)) {
        remote_boundary_data = remote_boundary_data.project_to_mortar(
            neighbor_face_mesh, neighbor_mortar_mesh, neighbor_mortar_size);
      }
      if (not neighbor_orientation.is_aligned()) {
        remote_boundary_data.orient_on_slice(
            neighbor_mortar_mesh.extents(), direction_from_neighbor.dimension(),
            neighbor_orientation.inverse_map());
      }
    }

    // Compute boundary contribution and add to operator
    apply_boundary_contribution(
        make_not_null(&result), std::move(local_boundary_data),
        std::move(remote_boundary_data), magnitude_of_face_normal,
        dg_element.mesh, mortar_id, mortar_mesh, mortar_size);
  }
  return result;
}

}  // namespace TestHelpers::elliptic::dg
