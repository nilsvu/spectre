// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Element.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"

namespace TestHelpers {
namespace elliptic {
namespace dg {

template <size_t Dim>
struct DgElement {
  Mesh<Dim> mesh;
  Element<Dim> element;
  ElementMap<Dim, Frame::Inertial> element_map;
  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
      inv_jacobian;
};

template <size_t Dim>
struct ElementOrdering {
  bool operator()(const ElementId<Dim>& lhs, const ElementId<Dim>& rhs) const
      noexcept {
    if (lhs.block_id() != rhs.block_id()) {
      return lhs.block_id() < rhs.block_id();
    }
    for (size_t d = 0; d < Dim; d++) {
      const size_t lhs_index = lhs.segment_ids().at(d).index();
      const size_t rhs_index = rhs.segment_ids().at(d).index();
      if (lhs_index != rhs_index) {
        return lhs_index < rhs_index;
      }
    }
    return false;
  }
};

template <size_t Dim>
using DgElementArray =
    std::map<ElementId<Dim>, DgElement<Dim>, ElementOrdering<Dim>>;

template <size_t Dim>
DgElementArray<Dim> make_elements(
    const DomainCreator<Dim>& domain_creator) noexcept;

template <size_t VolumeDim>
using MortarId = std::pair<::Direction<VolumeDim>, ElementId<VolumeDim>>;
template <size_t VolumeDim, typename ValueType>
using MortarMap = std::unordered_map<MortarId<VolumeDim>, ValueType,
                                     boost::hash<MortarId<VolumeDim>>>;
template <size_t MortarDim>
using MortarSizes = std::array<Spectral::MortarSize, MortarDim>;

template <size_t VolumeDim>
MortarMap<VolumeDim, std::pair<Mesh<VolumeDim - 1>, MortarSizes<VolumeDim - 1>>>
make_mortars(const ElementId<VolumeDim>& element_id,
             const DgElementArray<VolumeDim>& dg_elements) noexcept;

template <typename System, typename TagsList, typename PackageBoundaryData,
          typename ApplyBoundaryContribution, size_t Dim = System::volume_dim,
          typename PrimalFields = typename System::primal_fields,
          typename AuxiliaryFields = typename System::auxiliary_fields,
          typename FluxesComputer = typename System::fluxes,
          typename SourcesComputer = typename System::sources>
Variables<TagsList> apply_dg_operator(
    const ElementId<Dim>& element_id, const DgElementArray<Dim>& dg_elements,
    const std::unordered_map<ElementId<Dim>, Variables<TagsList>>& workspace,
    const FluxesComputer& fluxes_computer,
    PackageBoundaryData&& package_boundary_data,
    ApplyBoundaryContribution&& apply_boundary_contribution) {
  static constexpr size_t volume_dim = Dim;
  using Vars = Variables<TagsList>;

  const auto& dg_element = dg_elements.at(element_id);
  const auto& vars = workspace.at(element_id);

  const size_t num_points = dg_element.mesh.number_of_grid_points();
  Vars result{num_points};

  // Compute fluxes
  const auto fluxes =
      ::elliptic::first_order_fluxes<volume_dim, PrimalFields, AuxiliaryFields>(
          vars, fluxes_computer);

  // Compute divergences
  const auto div_fluxes =
      divergence(fluxes, dg_element.mesh, dg_element.inv_jacobian);

  // Compute bulk contribution in central element
  ::elliptic::first_order_operator(
      make_not_null(&result), div_fluxes,
      ::elliptic::first_order_sources<PrimalFields, AuxiliaryFields,
                                      SourcesComputer>(vars));

  // Setup mortars
  const auto mortars = make_mortars(element_id, dg_elements);

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
    // Assuming Euclidean magnitude. Could retrieve the magnitude compute tag
    // from the `system` but then we need to handle its arguments.
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
    const auto local_boundary_data = package_boundary_data(
        face_normal, normal_dot_fluxes, div_fluxes_on_face);

    // Assemble remote boundary data
    auto remote_face_normal = face_normal;
    for (size_t d = 0; d < volume_dim; d++) {
      remote_face_normal.get(d) *= -1.;
    }
    std::decay_t<decltype(local_boundary_data)> remote_boundary_data;
    if (neighbor_id == ElementId<volume_dim>::external_boundary_id()) {
      // On exterior ("ghost") faces, manufacture boundary data that represent
      // homogeneous Dirichlet boundary conditions
      const auto vars_on_face = data_on_slice(vars, dg_element.mesh.extents(),
                                              dimension, slice_index);
      Vars ghost_vars{face_num_points};
      ::elliptic::dg::homogeneous_dirichlet_boundary_conditions<PrimalFields>(
          make_not_null(&ghost_vars), vars_on_face);
      const auto ghost_fluxes =
          ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                         AuxiliaryFields>(ghost_vars,
                                                          fluxes_computer);
      const auto ghost_normal_dot_fluxes =
          normal_dot_flux<TagsList>(remote_face_normal, ghost_fluxes);
      remote_boundary_data =
          package_boundary_data(remote_face_normal, ghost_normal_dot_fluxes,
                                // TODO: Is this correct?
                                div_fluxes_on_face);
    } else {
      // On internal boundaries, get neighbor data from workspace
      const auto& neighbor_orientation =
          dg_element.element.neighbors().at(direction).orientation();
      const auto direction_from_neighbor =
          neighbor_orientation(direction.opposite());
      const auto& neighbor = dg_elements.at(neighbor_id);
      const auto& remote_vars = workspace.at(neighbor_id);
      // TODO: Make sure fluxes args are used from neighbor
      const auto remote_fluxes =
          ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                         AuxiliaryFields>(remote_vars,
                                                          fluxes_computer);
      const auto remote_div_fluxes_on_face = data_on_slice(
          divergence(remote_fluxes, neighbor.mesh, neighbor.inv_jacobian),
          neighbor.mesh.extents(), direction_from_neighbor.dimension(),
          index_to_slice_at(neighbor.mesh.extents(), direction_from_neighbor));
      const auto remote_fluxes_on_face = data_on_slice(
          remote_fluxes, neighbor.mesh.extents(),
          direction_from_neighbor.dimension(),
          index_to_slice_at(neighbor.mesh.extents(), direction_from_neighbor));
      auto remote_normal_dot_fluxes =
          normal_dot_flux<TagsList>(remote_face_normal, remote_fluxes_on_face);
      remote_boundary_data =
          package_boundary_data(remote_face_normal, remote_normal_dot_fluxes,
                                remote_div_fluxes_on_face);
      // TODO: orient and project
    }

    // Compute boundary contribution and add to operator
    apply_boundary_contribution(make_not_null(&result), local_boundary_data,
                                remote_boundary_data, magnitude_of_face_normal,
                                dg_element.mesh, mortar_id, mortar_mesh,
                                mortar_size);
  }
  return result;
}

template <typename System, typename PackageBoundaryData,
          typename ApplyBoundaryContribution>
DenseMatrix<double> build_operator_matrix(
    const DomainCreator<System::volume_dim>& domain_creator,
    const typename System::fluxes& fluxes_computer,
    PackageBoundaryData&& package_boundary_data,
    ApplyBoundaryContribution&& apply_boundary_contribution) {
  static constexpr size_t volume_dim = System::volume_dim;
  using Vars = db::item_type<typename System::fields_tag>;

  const auto elements = make_elements(domain_creator);

  // Create workspace vars for each element and count full operator size
  std::unordered_map<ElementId<volume_dim>, Vars> workspace{};
  size_t operator_size = 0;
  for (const auto& id_and_element : elements) {
    Vars element_data{id_and_element.second.mesh.number_of_grid_points()};
    operator_size += element_data.size();
    workspace[id_and_element.first] = std::move(element_data);
  }

  DenseMatrix<double> operator_matrix{operator_size, operator_size};
  // Build the matrix by applying the operator to unit vectors
  size_t i_across_elements = 0;
  size_t j_across_elements = 0;
  for (const auto& active_id_and_element : elements) {
    const size_t size_active_element =
        workspace.at(active_id_and_element.first).size();

    for (size_t i = 0; i < size_active_element; i++) {
      // Construct a unit vector
      for (const auto& id_and_element : elements) {
        auto& vars = workspace.at(id_and_element.first);
        vars = Vars{workspace.at(id_and_element.first).number_of_grid_points(),
                    0.};
        if (id_and_element.first == active_id_and_element.first) {
          vars.data()[i] = 1.;
        }
      }

      for (const auto& id_and_element : elements) {
        // Apply the operator
        const auto column_element_data = apply_dg_operator<System>(
            id_and_element.first, elements, workspace, fluxes_computer,
            package_boundary_data, apply_boundary_contribution);

        // Store result in matrix
        for (size_t j = 0; j < column_element_data.size(); j++) {
          operator_matrix(j_across_elements, i_across_elements) =
              column_element_data.data()[j];
          j_across_elements++;
        }
      }
      i_across_elements++;
      j_across_elements = 0;
    }
  }
  return operator_matrix;
}

}  // namespace dg
}  // namespace elliptic
}  // namespace TestHelpers
