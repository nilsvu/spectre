// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/OverlapData.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainHelpers.hpp"
#include "Utilities/TMPL.hpp"

#include "ParallelAlgorithms/LinearSolver/Tags.hpp"

// #include "Parallel/Printf.hpp"

namespace elliptic {
namespace dg {

namespace SubdomainOperator_detail {
// These functions are specific to the strong first-order internal penalty
// scheme
template <typename BoundaryData, size_t Dim,
          typename NumericalFluxesComputerType, typename FluxesComputerType,
          typename NormalDotFluxesTags, typename DivFluxesTags,
          typename... AuxiliaryFields>
BoundaryData package_boundary_data(
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const FluxesComputerType& fluxes_computer, const Mesh<Dim - 1>& face_mesh,
    const tnsr::i<DataVector, Dim>& face_normal,
    const Variables<NormalDotFluxesTags>& n_dot_fluxes,
    const Variables<DivFluxesTags>& div_fluxes,
    tmpl::list<AuxiliaryFields...> /*meta*/) noexcept {
  return ::dg::FirstOrderScheme::package_boundary_data(
      numerical_fluxes_computer, face_mesh, n_dot_fluxes,
      get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_fluxes)...,
      get<::Tags::div<
          ::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>, Frame::Inertial>>>(
          div_fluxes)...,
      face_normal, fluxes_computer);
}
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

template <typename PrimalFields, typename AuxiliaryFields, size_t Dim,
          typename BoundaryData, typename FluxesComputerType,
          typename NumericalFluxesComputerType,
          typename FieldsTags = tmpl::append<PrimalFields, AuxiliaryFields>,
          typename FluxesTags = db::wrap_tags_in<
              ::Tags::Flux, FieldsTags, tmpl::size_t<Dim>, Frame::Inertial>,
          typename DivFluxesTags = db::wrap_tags_in<::Tags::div, FluxesTags>>
void exterior_boundary_data(
    const gsl::not_null<BoundaryData*> boundary_data,
    const Variables<FieldsTags>& vars_on_interior_face,
    const Variables<DivFluxesTags>& div_fluxes_on_interior_face,
    const Mesh<Dim - 1>& face_mesh,
    const tnsr::i<DataVector, Dim>& interior_face_normal,
    const FluxesComputerType& fluxes_computer,
    const NumericalFluxesComputerType& numerical_fluxes_computer) noexcept {
  static constexpr size_t volume_dim = Dim;
  // On exterior ("ghost") faces, manufacture boundary data that represent
  // homogeneous Dirichlet boundary conditions
  Variables<FieldsTags> ghost_vars{
      vars_on_interior_face.number_of_grid_points()};
  ::elliptic::dg::homogeneous_dirichlet_boundary_conditions<PrimalFields>(
      make_not_null(&ghost_vars), vars_on_interior_face);
  const auto ghost_fluxes =
      ::elliptic::first_order_fluxes<volume_dim, PrimalFields, AuxiliaryFields>(
          ghost_vars, fluxes_computer);
  auto exterior_face_normal = interior_face_normal;
  for (size_t d = 0; d < volume_dim; d++) {
    exterior_face_normal.get(d) *= -1.;
  }
  const auto ghost_normal_dot_fluxes =
      normal_dot_flux<FieldsTags>(exterior_face_normal, ghost_fluxes);
  *boundary_data = package_boundary_data<BoundaryData>(
      numerical_fluxes_computer, fluxes_computer, face_mesh,
      exterior_face_normal, ghost_normal_dot_fluxes,
      // TODO: Is this correct?
      div_fluxes_on_interior_face, AuxiliaryFields{});
}

template <size_t Dim>
std::pair<tnsr::i<DataVector, Dim>, Scalar<DataVector>>
face_normal_and_magnitude(const Mesh<Dim - 1>& face_mesh,
                          const ElementMap<Dim, Frame::Inertial>& element_map,
                          const Direction<Dim>& direction) noexcept {
  auto face_normal =
      unnormalized_face_normal(face_mesh, element_map, direction);
  // TODO: handle curved backgrounds
  auto magnitude_of_face_normal = magnitude(face_normal);
  for (size_t d = 0; d < Dim; d++) {
    face_normal.get(d) /= get(magnitude_of_face_normal);
  }
  return {std::move(face_normal), std::move(magnitude_of_face_normal)};
}

}  // namespace SubdomainOperator_detail

template <typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputerType, size_t Dim, typename FluxesComputerType,
          typename NumericalFluxesComputerType, typename SubdomainDataType,
          typename AllFieldsTags = tmpl::append<PrimalFields, AuxiliaryFields>>
static SubdomainDataType apply_subdomain_operator(
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inv_jacobian,
    const FluxesComputerType& fluxes_computer,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const db::const_item_type<domain::Tags::Interface<
        domain::Tags::InternalDirections<Dim>,
        ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>&
        internal_face_normals,
    const db::const_item_type<domain::Tags::Interface<
        domain::Tags::BoundaryDirectionsInterior<Dim>,
        ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>&
        boundary_face_normals,
    const db::const_item_type<domain::Tags::Interface<
        domain::Tags::InternalDirections<Dim>,
        ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>&
        internal_face_normal_magnitudes,
    const db::const_item_type<domain::Tags::Interface<
        domain::Tags::BoundaryDirectionsInterior<Dim>,
        ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>&
        boundary_face_normal_magnitudes,
    const db::const_item_type<
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>& mortar_meshes,
    const db::const_item_type<
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
    const SubdomainDataType& arg) noexcept {
  static constexpr size_t volume_dim = Dim;
  using all_fields_tags = AllFieldsTags;
  using fluxes_tags =
      db::wrap_tags_in<::Tags::Flux, all_fields_tags, tmpl::size_t<volume_dim>,
                       Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;
  using BoundaryData = ::dg::SimpleBoundaryData<
      tmpl::remove_duplicates<tmpl::append<
          n_dot_fluxes_tags,
          typename NumericalFluxesComputerType::package_field_tags>>,
      typename NumericalFluxesComputerType::package_extra_tags>;

  using OverlapDataType = typename SubdomainDataType::overlap_data_type;

  SubdomainDataType result{arg.element_data.number_of_grid_points()};
  // Since the subdomain operator is called repeatedly for the subdomain solve
  // it could help performance to avoid re-allocating memory by storing the
  // tensor quantities in a buffer.
//   Parallel::printf("\n\nComputing subdomain operator of:\n%s\n",
//                    arg.element_data);
  // Compute bulk contribution in central element
  const auto central_fluxes =
      elliptic::first_order_fluxes<volume_dim, PrimalFields, AuxiliaryFields>(
          arg.element_data, fluxes_computer);
  const auto central_div_fluxes =
      divergence(central_fluxes, mesh, inv_jacobian);
  elliptic::first_order_operator(
      make_not_null(&result.element_data), central_div_fluxes,
      elliptic::first_order_sources<PrimalFields, AuxiliaryFields,
                                    SourcesComputerType>(arg.element_data));
  // Add boundary contributions
  for (const auto& mortar_id_and_mesh : mortar_meshes) {
    const auto& mortar_id = mortar_id_and_mesh.first;
    const auto& mortar_mesh = mortar_id_and_mesh.second;
    const auto& mortar_size = mortar_sizes.at(mortar_id);
    const auto& direction = mortar_id.first;
    const auto& neighbor_id = mortar_id.second;

    const size_t dimension = direction.dimension();
    const auto face_mesh = mesh.slice_away(dimension);
    const size_t slice_index = index_to_slice_at(mesh.extents(), direction);

    const bool is_boundary =
        neighbor_id == ElementId<volume_dim>::external_boundary_id();

    const tnsr::i<DataVector, volume_dim>& face_normal =
        is_boundary ? boundary_face_normals.at(direction)
                    : internal_face_normals.at(direction);
    const Scalar<DataVector>& magnitude_of_face_normal =
        is_boundary ? boundary_face_normal_magnitudes.at(direction)
                    : internal_face_normal_magnitudes.at(direction);

    // Compute normal dot fluxes
    const auto central_fluxes_on_face =
        data_on_slice(central_fluxes, mesh.extents(), dimension, slice_index);
    const auto normal_dot_central_fluxes =
        normal_dot_flux<all_fields_tags>(face_normal, central_fluxes_on_face);

    // Slice flux divergences to face
    const auto central_div_fluxes_on_face = data_on_slice(
        central_div_fluxes, mesh.extents(), dimension, slice_index);

    // Assemble local boundary data
    auto local_boundary_data =
        SubdomainOperator_detail::package_boundary_data<BoundaryData>(
            numerical_fluxes_computer, fluxes_computer, face_mesh, face_normal,
            normal_dot_central_fluxes, central_div_fluxes_on_face,
            AuxiliaryFields{});
    if (::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
      local_boundary_data = local_boundary_data.project_to_mortar(
          face_mesh, mortar_mesh, mortar_size);
    }

    // Assemble remote boundary data
    BoundaryData remote_boundary_data;
    if (is_boundary) {
      const auto central_vars_on_face = data_on_slice(
          arg.element_data, mesh.extents(), dimension, slice_index);
      SubdomainOperator_detail::exterior_boundary_data<PrimalFields,
                                                       AuxiliaryFields>(
          make_not_null(&remote_boundary_data), central_vars_on_face,
          central_div_fluxes_on_face, face_mesh, face_normal, fluxes_computer,
          numerical_fluxes_computer);
      // No projections necessary since exterior mortars cover the full face
    } else {
      // On internal boundaries, get neighbor data from the operand.
      // Note that all overlap data must have been oriented to the perspective
      // of the central element at this point.
      const auto direction_from_neighbor = direction.opposite();
      // Parallel::printf("> Internal mortar %s:\n", mortar_id);
      const auto& overlap_data = arg.boundary_data.at(mortar_id);
      ASSERT(overlap_data.direction == direction_from_neighbor,
             "Directions mismatch. Did you forget to orient the overlap data?");
      // Parallel::printf("Overlap data: %s\n", overlap_data.field_data);
      const auto& neighbor_mesh = overlap_data.volume_mesh;
      auto neighbor_face_mesh = neighbor_mesh.slice_away(dimension);
      size_t neighbor_face_slice_index =
          index_to_slice_at(neighbor_mesh.extents(), direction_from_neighbor);
    //   Parallel::printf("slice index: %d\n", neighbor_face_slice_index);

      // Extend the overlap data to the full neighbor mesh by filling it
      // with zeros and adding the overlapping slices
      const auto neighbor_data = overlap_data.extended_field_data();
    //   Parallel::printf("Extended overlap data: %s\n", neighbor_data);

      // Compute the volume contribution in the neighbor from the extended
      // overlap data
      // TODO: Make sure fluxes args are used from neighbor
      const auto neighbor_fluxes =
          ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                         AuxiliaryFields>(neighbor_data,
                                                          fluxes_computer);
      const auto neighbor_logical_coords = logical_coordinates(neighbor_mesh);
      const auto neighbor_inv_jacobian =
          overlap_data.element_map.inv_jacobian(neighbor_logical_coords);
    //   Parallel::printf("Inv jac for %s: %s\n", neighbor_id,
    //                    neighbor_inv_jacobian);
      const auto neighbor_div_fluxes =
          divergence(neighbor_fluxes, neighbor_mesh, neighbor_inv_jacobian);
      typename SubdomainDataType::element_data_type neighbor_result_extended{
          neighbor_mesh.number_of_grid_points()};
      elliptic::first_order_operator(
          make_not_null(&neighbor_result_extended), neighbor_div_fluxes,
          elliptic::first_order_sources<PrimalFields, AuxiliaryFields,
                                        SourcesComputerType>(neighbor_data));
    //   Parallel::printf("Extended result on overlap: %s\n",
    //                    neighbor_result_extended);

      auto neighbor_face_normal_and_magnitude =
          SubdomainOperator_detail::face_normal_and_magnitude(
              neighbor_face_mesh, overlap_data.element_map,
              direction_from_neighbor);
      for (size_t d = 0; d < volume_dim; d++) {
        ASSERT(neighbor_face_normal_and_magnitude.first.get(d) ==
                   -1. * face_normal.get(d),
               "Face normals should be opposite");
      }
      ASSERT(get(neighbor_face_normal_and_magnitude.second) ==
                 get(magnitude_of_face_normal),
             "Face normals magnitudes should be the same");

      auto neighbor_fluxes_on_face =
          data_on_slice(neighbor_fluxes, neighbor_mesh.extents(), dimension,
                        neighbor_face_slice_index);
      auto neighbor_div_fluxes_on_face =
          data_on_slice(neighbor_div_fluxes, neighbor_mesh.extents(), dimension,
                        neighbor_face_slice_index);
      auto neighbor_normal_dot_fluxes = normal_dot_flux<all_fields_tags>(
          neighbor_face_normal_and_magnitude.first, neighbor_fluxes_on_face);
      remote_boundary_data =
          SubdomainOperator_detail::package_boundary_data<BoundaryData>(
              numerical_fluxes_computer, fluxes_computer, neighbor_face_mesh,
              neighbor_face_normal_and_magnitude.first,
              neighbor_normal_dot_fluxes, neighbor_div_fluxes_on_face,
              AuxiliaryFields{});
      if (::dg::needs_projection(neighbor_face_mesh, mortar_mesh,
                                 mortar_size)) {
        remote_boundary_data = remote_boundary_data.project_to_mortar(
            neighbor_face_mesh, mortar_mesh, mortar_size);
      }

      // Apply the boundary contribution to the neighbor overlap
      SubdomainOperator_detail::apply_boundary_contribution(
          make_not_null(&neighbor_result_extended), numerical_fluxes_computer,
          remote_boundary_data, local_boundary_data,
          neighbor_face_normal_and_magnitude.second, neighbor_mesh,
          direction_from_neighbor, mortar_mesh, mortar_size);
    //   Parallel::printf(
    //       "Extended result on overlap incl. boundary contrib from central "
    //       "element: %s\n",
    //       neighbor_result_extended);

      // Add boundary contributions from the neighbor's neighbors to the
      // extended overlap data. We need only consider faces that share points
      // with the overlap region.
      for (const auto& neighbor_mortar_id_and_mesh :
           overlap_data.perpendicular_mortar_meshes) {
        const auto& neighbor_mortar_id = neighbor_mortar_id_and_mesh.first;
        const auto& neighbor_face_direction = neighbor_mortar_id.first;
        if (neighbor_face_direction == direction_from_neighbor) {
          continue;
        }
        const auto& neighbors_neighbor_id = neighbor_mortar_id.second;
        const bool neighbor_face_is_boundary =
            neighbors_neighbor_id ==
            ElementId<volume_dim>::external_boundary_id();
        const size_t neighbor_face_dimension =
            neighbor_face_direction.dimension();
        neighbor_face_slice_index =
            index_to_slice_at(neighbor_mesh.extents(), neighbor_face_direction);
        neighbor_face_mesh = neighbor_mesh.slice_away(neighbor_face_dimension);
        const auto& neighbor_face_num_points =
            neighbor_face_mesh.number_of_grid_points();
        const auto& neighbor_mortar_mesh = neighbor_mortar_id_and_mesh.second;
        const auto& neighbor_mortar_size =
            overlap_data.perpendicular_mortar_sizes.at(neighbor_mortar_id);

        neighbor_face_normal_and_magnitude =
            SubdomainOperator_detail::face_normal_and_magnitude(
                neighbor_face_mesh, overlap_data.element_map,
                neighbor_face_direction);

        neighbor_fluxes_on_face =
            data_on_slice(neighbor_fluxes, neighbor_mesh.extents(),
                          neighbor_face_dimension, neighbor_face_slice_index);
        neighbor_div_fluxes_on_face =
            data_on_slice(neighbor_div_fluxes, neighbor_mesh.extents(),
                          neighbor_face_dimension, neighbor_face_slice_index);
        neighbor_normal_dot_fluxes = normal_dot_flux<all_fields_tags>(
            neighbor_face_normal_and_magnitude.first, neighbor_fluxes_on_face);
        auto neighbor_local_boundary_data =
            SubdomainOperator_detail::package_boundary_data<BoundaryData>(
                numerical_fluxes_computer, fluxes_computer, neighbor_face_mesh,
                neighbor_face_normal_and_magnitude.first,
                neighbor_normal_dot_fluxes, neighbor_div_fluxes_on_face,
                AuxiliaryFields{});
        // BoundaryData neighbor_local_boundary_data;
        // SubdomainOperator_detail::interior_boundary_data<PrimalFields,
        //                                                  AuxiliaryFields>(
        //     make_not_null(&neighbor_local_boundary_data), neighbor_mesh,
        //     neighbor_fluxes, neighbor_div_fluxes, neighbor_face_direction,
        //     neighbor_face_normal_and_magnitude.first, fluxes_computer,
        //     numerical_fluxes_computer);
        if (::dg::needs_projection(neighbor_face_mesh, neighbor_mortar_mesh,
                                   neighbor_mortar_size)) {
          neighbor_local_boundary_data =
              neighbor_local_boundary_data.project_to_mortar(
                  neighbor_face_mesh, neighbor_mortar_mesh,
                  neighbor_mortar_size);
        }

        BoundaryData neighbor_remote_boundary_data;
        if (neighbor_face_is_boundary) {
          const auto neighbor_face_data =
              data_on_slice(neighbor_data, neighbor_mesh.extents(),
                            neighbor_face_dimension, neighbor_face_slice_index);
          SubdomainOperator_detail::exterior_boundary_data<PrimalFields,
                                                           AuxiliaryFields>(
              make_not_null(&neighbor_remote_boundary_data), neighbor_face_data,
              neighbor_div_fluxes_on_face, neighbor_face_mesh,
              neighbor_face_normal_and_magnitude.first, fluxes_computer,
              numerical_fluxes_computer);
        } else {
          // Assume the data on the neighbor's neighbor is zero.

          // TODO: Make sure this works with h-refinement.. we know the
          // data on mortars to other neighbors in the same direction, so it
          // shouldn't be zero.

          // The normal is probably irrelevent in this case, but we compute it
          // to be safe.
          auto neighbor_remote_face_normal =
              neighbor_face_normal_and_magnitude.first;
          for (size_t d = 0; d < volume_dim; d++) {
            neighbor_remote_face_normal.get(d) *= -1.;
          }
          neighbor_remote_boundary_data =
              SubdomainOperator_detail::package_boundary_data<BoundaryData>(
                  numerical_fluxes_computer, fluxes_computer,
                  neighbor_face_mesh, neighbor_remote_face_normal,
                  Variables<n_dot_fluxes_tags>{neighbor_face_num_points, 0.},
                  Variables<div_fluxes_tags>{neighbor_face_num_points, 0.},
                  AuxiliaryFields{});
        }
        SubdomainOperator_detail::apply_boundary_contribution(
            make_not_null(&neighbor_result_extended), numerical_fluxes_computer,
            neighbor_local_boundary_data, neighbor_remote_boundary_data,
            neighbor_face_normal_and_magnitude.second, neighbor_mesh,
            neighbor_face_direction, neighbor_mortar_mesh,
            neighbor_mortar_size);
      }
    //   Parallel::printf(
    //       "Extended result on overlap incl. all boundary contribs: %s\n",
    //       neighbor_result_extended);

      // Take only the part of the neighbor data that lies within the overlap
      const auto neighbor_result =
          LinearSolver::schwarz_detail::data_on_overlap(
              neighbor_result_extended, neighbor_mesh.extents(),
              overlap_data.overlap_extents, direction_from_neighbor);
      // TODO: Fake boundary contributions from the other mortars of the
      // neighbor by filling their data with zeros
    //   Parallel::printf("Final result on overlap: %s\n", neighbor_result);

      // Construct the data that represents the subdomain operator applied to
      // the overlap. We make things easy by copying the operand and changing
      // the field data. This should be improved.
      OverlapDataType overlap_result = overlap_data;
      overlap_result.field_data = std::move(neighbor_result);
      result.boundary_data.emplace(mortar_id, std::move(overlap_result));
    }

    // Apply the boundary contribution to the central element
    SubdomainOperator_detail::apply_boundary_contribution(
        make_not_null(&result.element_data), numerical_fluxes_computer,
        local_boundary_data, remote_boundary_data, magnitude_of_face_normal,
        mesh, direction, mortar_mesh, mortar_size);
  }

//   Parallel::printf("Result:\n%s\n", result.element_data);
  return result;
}

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename FluxesComputerTag, typename SourcesComputer,
          typename NumericalFluxesComputerTag, typename OptionsGroup>
struct SubdomainOperator {
 private:
  using all_fields_tags = tmpl::append<PrimalFields, AuxiliaryFields>;

 public:
  static constexpr size_t volume_dim = Dim;
  using SubdomainDataType = LinearSolver::schwarz_detail::SubdomainData<
      Dim, Variables<all_fields_tags>,
      SubdomainOperator_detail::OverlapData<Dim, all_fields_tags>>;
  using collect_overlap_data =
      SubdomainOperator_detail::CollectOverlapData<Dim, all_fields_tags,
                                                   OptionsGroup>;

  using argument_tags = tmpl::list<
      domain::Tags::Mesh<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      FluxesComputerTag, NumericalFluxesComputerTag,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;
  static constexpr auto apply =
      &apply_subdomain_operator<PrimalFields, AuxiliaryFields, SourcesComputer,
                                Dim, db::const_item_type<FluxesComputerTag>,
                                db::const_item_type<NumericalFluxesComputerTag>,
                                SubdomainDataType, all_fields_tags>;
};

}  // namespace dg
}  // namespace elliptic
