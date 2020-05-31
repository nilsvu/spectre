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
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"
#include "Utilities/TupleFromTypelist.hpp"
#include "Utilities/TupleSlice.hpp"

#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

// #include "Parallel/Printf.hpp"

namespace elliptic {
namespace dg {

namespace SubdomainOperator_detail {
// These functions are specific to the strong first-order internal penalty
// scheme
template <typename BoundaryData, size_t Dim,
          typename NumericalFluxesComputerType, typename FluxesComputerType,
          typename NormalDotFluxesTags, typename DivFluxesTags,
          typename... FluxesArgs, typename... AuxiliaryFields>
BoundaryData package_boundary_data(
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const FluxesComputerType& fluxes_computer, const Mesh<Dim - 1>& face_mesh,
    const tnsr::i<DataVector, Dim>& face_normal,
    const Variables<NormalDotFluxesTags>& n_dot_fluxes,
    const Variables<DivFluxesTags>& div_fluxes,
    const std::tuple<FluxesArgs...>& fluxes_args,
    tmpl::list<AuxiliaryFields...> /*meta*/) noexcept {
  return std::apply(
      [&numerical_fluxes_computer, &face_mesh, &n_dot_fluxes, &div_fluxes,
       &face_normal, &fluxes_computer](const auto&... expanded_fluxes_args) {
        return ::dg::FirstOrderScheme::package_boundary_data(
            numerical_fluxes_computer, face_mesh, n_dot_fluxes,
            get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_fluxes)...,
            get<::Tags::div<::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>,
                                         Frame::Inertial>>>(div_fluxes)...,
            face_normal, fluxes_computer, expanded_fluxes_args...);
      },
      fluxes_args);
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
          typename NumericalFluxesComputerType, typename... FluxesArgs,
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
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const std::tuple<FluxesArgs...>& fluxes_args) noexcept {
  static constexpr size_t volume_dim = Dim;
  // On exterior ("ghost") faces, manufacture boundary data that represent
  // homogeneous Dirichlet boundary conditions
  Variables<FieldsTags> ghost_vars{
      vars_on_interior_face.number_of_grid_points()};
  ::elliptic::dg::homogeneous_dirichlet_boundary_conditions<PrimalFields>(
      make_not_null(&ghost_vars), vars_on_interior_face);
  const auto ghost_fluxes = std::apply(
      [&ghost_vars, &fluxes_computer](const auto&... expanded_fluxes_args) {
        return ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                              AuxiliaryFields>(
            ghost_vars, fluxes_computer, expanded_fluxes_args...);
      },
      fluxes_args);
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
      div_fluxes_on_interior_face, fluxes_args, AuxiliaryFields{});
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

// By default don't do any slicing
template <typename Arg>
struct slice_arg_to_face_impl {
  template <size_t Dim>
  static Arg apply(const Arg& arg, const Index<Dim>& /*extents*/,
                   const Direction<Dim>& /*direction*/) noexcept {
    return arg;
  }
};
// Slice tensors to the face
template <typename DataType, typename Symm, typename IndexList>
struct slice_arg_to_face_impl<Tensor<DataType, Symm, IndexList>> {
  struct TempTag : db::SimpleTag {
    using type = Tensor<DataType, Symm, IndexList>;
  };
  template <size_t Dim>
  static Tensor<DataType, Symm, IndexList> apply(
      const Tensor<DataType, Symm, IndexList>& arg, const Index<Dim>& extents,
      const Direction<Dim>& direction) noexcept {
    Variables<tmpl::list<TempTag>> temp_vars{arg.begin()->size()};
    get<TempTag>(temp_vars) = arg;
    return get<TempTag>(data_on_slice(temp_vars, extents, direction.dimension(),
                                      index_to_slice_at(extents, direction)));
  }
};

// Slice fluxes args to faces. Used on overlap faces perpendicular to the
// subdomain interface.
// TODO: find a way to avoid this.
template <size_t Dim, typename Arg>
auto slice_arg_to_face(const Arg& arg, const Index<Dim>& extents,
                       const Direction<Dim>& direction) noexcept {
  return slice_arg_to_face_impl<Arg>::apply(arg, extents, direction);
}

CREATE_IS_CALLABLE(at);
CREATE_IS_CALLABLE_V(at);

template <
    size_t Dim, typename Arg,
    Requires<is_at_callable_v<Arg, LinearSolver::Schwarz::OverlapId<Dim>>> =
        nullptr>
decltype(auto) unmap_overlap_arg(
    const Arg& arg,
    const LinearSolver::Schwarz::OverlapId<Dim>& overlap_id) noexcept {
  return arg.at(overlap_id);
}

template <
    size_t Dim, typename Arg,
    Requires<not is_at_callable_v<Arg, LinearSolver::Schwarz::OverlapId<Dim>>> =
        nullptr>
decltype(auto) unmap_overlap_arg(
    const Arg& arg,
    const LinearSolver::Schwarz::OverlapId<Dim>& /*overlap_id*/) noexcept {
  return arg;
}

}  // namespace SubdomainOperator_detail

// Compute bulk contribution in central element
template <typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer, typename ResultTagsList,
          typename ArgTagsList, typename FluxesComputer, size_t Dim,
          typename... FluxesArgs, typename... SourcesArgs,
          typename AllFields = tmpl::append<PrimalFields, AuxiliaryFields>,
          typename FluxesTags = db::wrap_tags_in<
              ::Tags::Flux, AllFields, tmpl::size_t<Dim>, Frame::Inertial>>
void apply_subdomain_center_volume(
    const gsl::not_null<Variables<ResultTagsList>*> result_element_data,
    const gsl::not_null<Variables<FluxesTags>*> fluxes,
    const gsl::not_null<Variables<db::wrap_tags_in<::Tags::div, FluxesTags>>*>
        div_fluxes,
    const FluxesComputer& fluxes_computer, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inv_jacobian,
    const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<SourcesArgs...>& sources_args,
    const Variables<ArgTagsList>& arg_element_data) noexcept {
  // Compute volume fluxes
  std::apply(
      [&fluxes, &arg_element_data,
       &fluxes_computer](const auto&... expanded_fluxes_args) {
        elliptic::first_order_fluxes<Dim, PrimalFields, AuxiliaryFields>(
            fluxes, Variables<AllFields>(arg_element_data), fluxes_computer,
            expanded_fluxes_args...);
      },
      fluxes_args);
  // Compute divergence of volume fluxes
  *div_fluxes = divergence(*fluxes, mesh, inv_jacobian);
  // Compute volume sources
  auto sources = std::apply(
      [&arg_element_data](const auto&... expanded_sources_args) {
        return elliptic::first_order_sources<PrimalFields, AuxiliaryFields,
                                             SourcesComputer>(
            Variables<AllFields>(arg_element_data), expanded_sources_args...);
      },
      sources_args);
  elliptic::first_order_operator(result_element_data, *div_fluxes,
                                 std::move(sources));
}

// Add boundary contributions
template <typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputerType, size_t Dim, typename FluxesComputerType,
          typename NumericalFluxesComputerType, typename... FluxesArgs,
          typename... OverlapFluxesArgs, typename... OverlapSourcesArgs,
          typename ResultTags, typename ArgTags,
          typename AllFieldsTags = tmpl::append<PrimalFields, AuxiliaryFields>>
static void apply_subdomain_face(
    const gsl::not_null<
        LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ResultTags>*>
        result,
    const Element<Dim>& element, const Mesh<Dim>& mesh,
    const FluxesComputerType& fluxes_computer,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const Direction<Dim>& direction,
    const tnsr::i<DataVector, Dim>& face_normal,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const db::const_item_type<
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>& mortar_meshes,
    const db::const_item_type<
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
    const LinearSolver::Schwarz::OverlapMap<Dim, Index<Dim>>&
        all_overlap_extents,
    const LinearSolver::Schwarz::OverlapMap<Dim, Mesh<Dim>>& all_overlap_meshes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ElementMap<Dim, Frame::Inertial>>& all_overlap_element_maps,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, Mesh<Dim - 1>>>& all_overlap_mortar_meshes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>&
        all_overlap_mortar_sizes,
    const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<OverlapFluxesArgs...>& all_overlap_fluxes_args,
    const std::tuple<OverlapSourcesArgs...>& all_overlap_sources_args,
    const LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ArgTags>&
        arg,
    const Variables<db::wrap_tags_in<
        ::Tags::Flux, tmpl::append<PrimalFields, AuxiliaryFields>,
        tmpl::size_t<Dim>, Frame::Inertial>>& central_fluxes,
    const Variables<db::wrap_tags_in<
        ::Tags::div,
        db::wrap_tags_in<::Tags::Flux,
                         tmpl::append<PrimalFields, AuxiliaryFields>,
                         tmpl::size_t<Dim>, Frame::Inertial>>>&
        central_div_fluxes) noexcept {
  static constexpr size_t volume_dim = Dim;
  using SubdomainDataType =
      LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ResultTags>;
  using all_fields_tags = AllFieldsTags;
  using vars_tag = ::Tags::Variables<all_fields_tags>;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using n_dot_fluxes_tag = db::add_tag_prefix<::Tags::NormalDotFlux, vars_tag>;
  using BoundaryData =
      ::dg::FirstOrderScheme::BoundaryData<NumericalFluxesComputerType>;

  const size_t dimension = direction.dimension();
  const auto face_mesh = mesh.slice_away(dimension);
  const size_t slice_index = index_to_slice_at(mesh.extents(), direction);

  // Compute normal dot fluxes
  const auto central_fluxes_on_face =
      data_on_slice(central_fluxes, mesh.extents(), dimension, slice_index);
  const auto normal_dot_central_fluxes =
      normal_dot_flux<all_fields_tags>(face_normal, central_fluxes_on_face);

  // Slice flux divergences to face
  const auto central_div_fluxes_on_face =
      data_on_slice(central_div_fluxes, mesh.extents(), dimension, slice_index);

  // Assemble local boundary data
  const auto local_boundary_data_on_face =
      SubdomainOperator_detail::package_boundary_data<BoundaryData>(
          numerical_fluxes_computer, fluxes_computer, face_mesh, face_normal,
          normal_dot_central_fluxes, central_div_fluxes_on_face, fluxes_args,
          AuxiliaryFields{});

  // Iterate over mortars in this directions
  for (const auto& mortar_id_and_mesh : mortar_meshes) {
    const auto& mortar_id = mortar_id_and_mesh.first;
    const auto& mortar_mesh = mortar_id_and_mesh.second;
    const auto& mortar_size = mortar_sizes.at(mortar_id);
    const auto& local_direction = mortar_id.first;
    const auto& neighbor_id = mortar_id.second;
    if (local_direction != direction) {
      continue;
    }

    const bool is_boundary =
        neighbor_id == ElementId<volume_dim>::external_boundary_id();

    // Project local boundary data to mortar
    const auto local_boundary_data =
        ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
            ? local_boundary_data_on_face.project_to_mortar(
                  face_mesh, mortar_mesh, mortar_size)
            : local_boundary_data_on_face;

    // Assemble remote boundary data
    BoundaryData remote_boundary_data;
    if (is_boundary) {
      const auto central_vars_on_face =
          Variables<all_fields_tags>(data_on_slice(
              arg.element_data, mesh.extents(), dimension, slice_index));
      SubdomainOperator_detail::exterior_boundary_data<PrimalFields,
                                                       AuxiliaryFields>(
          make_not_null(&remote_boundary_data), central_vars_on_face,
          central_div_fluxes_on_face, face_mesh, face_normal, fluxes_computer,
          numerical_fluxes_computer, fluxes_args);
      // No projections necessary since exterior mortars cover the full face
    } else {
      // On internal boundaries, get neighbor data from the operand.
      // Note that data on overlaps is oriented according to the neighbor that
      // it is on, as is all geometric information that we have on the neighbor.
      // Only when data cross element boundaries do we need to re-orient.
      const auto& orientation = element.neighbors().at(direction).orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      const size_t dimension_in_neighbor = direction_from_neighbor.dimension();
      //   Parallel::printf("> Internal mortar %s:\n", mortar_id);
      const auto& overlap_data = arg.overlap_data.at(mortar_id);
      //   Parallel::printf("Overlap data: %s\n", overlap_data);
      const auto& neighbor_mesh = all_overlap_meshes.at(mortar_id);
      auto neighbor_face_mesh = neighbor_mesh.slice_away(dimension_in_neighbor);
      size_t neighbor_face_slice_index =
          index_to_slice_at(neighbor_mesh.extents(), direction_from_neighbor);
      // Parallel::printf("slice index: %d\n", neighbor_face_slice_index);

      // Extend the overlap data to the full neighbor mesh by filling it
      // with zeros and adding the overlapping slices
      const auto neighbor_data = Variables<all_fields_tags>(
          LinearSolver::Schwarz::extended_overlap_data(
              overlap_data, neighbor_mesh.extents(),
              all_overlap_extents.at(mortar_id), direction_from_neighbor));
      // Parallel::printf("Extended overlap data: %s\n", neighbor_data);

      // Compute the volume contribution in the neighbor from the extended
      // overlap data
      const auto neighbor_fluxes = std::apply(
          [&neighbor_data, &fluxes_computer,
           &mortar_id](const auto&... expanded_overlap_fluxes_args) noexcept {
            return ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                                  AuxiliaryFields>(
                neighbor_data, fluxes_computer,
                SubdomainOperator_detail::unmap_overlap_arg(
                    expanded_overlap_fluxes_args, mortar_id)...);
          },
          all_overlap_fluxes_args);
      // TODO: These could be cached in the databox
      const auto neighbor_logical_coords = logical_coordinates(neighbor_mesh);
      const auto neighbor_inv_jacobian =
          all_overlap_element_maps.at(mortar_id).inv_jacobian(
              neighbor_logical_coords);
      // Parallel::printf("Inv jac for %s: %s\n", neighbor_id,
      //                  neighbor_inv_jacobian);
      const auto neighbor_div_fluxes =
          divergence(neighbor_fluxes, neighbor_mesh, neighbor_inv_jacobian);
      typename SubdomainDataType::ElementData neighbor_result_extended{
          neighbor_mesh.number_of_grid_points()};
      auto neighbor_sources = std::apply(
          [&neighbor_data,
           &mortar_id](const auto&... expanded_overlap_sources_args) noexcept {
            return elliptic::first_order_sources<PrimalFields, AuxiliaryFields,
                                                 SourcesComputerType>(
                neighbor_data,
                SubdomainOperator_detail::unmap_overlap_arg(
                    expanded_overlap_sources_args, mortar_id)...);
          },
          all_overlap_sources_args);
      elliptic::first_order_operator(make_not_null(&neighbor_result_extended),
                                     neighbor_div_fluxes,
                                     std::move(neighbor_sources));
      // Parallel::printf("Extended result on overlap: %s\n",
      //                  neighbor_result_extended);

      auto neighbor_face_normal_and_magnitude =
          SubdomainOperator_detail::face_normal_and_magnitude(
              neighbor_face_mesh, all_overlap_element_maps.at(mortar_id),
              direction_from_neighbor);
      for (size_t d = 0; d < volume_dim; d++) {
        // Only holds for aligned elements
        // TODO: mirror transform the face_normal to the neighbor?
        // ASSERT(neighbor_face_normal_and_magnitude.first.get(d) ==
        //            -1. * face_normal.get(d),
        //        "Face normals should be opposite");
      }
      //   ASSERT(get(neighbor_face_normal_and_magnitude.second) ==
      //              get(magnitude_of_face_normal),
      //          "Face normals magnitudes should be the same");

      auto neighbor_fluxes_on_face =
          data_on_slice(neighbor_fluxes, neighbor_mesh.extents(),
                        dimension_in_neighbor, neighbor_face_slice_index);
      auto neighbor_div_fluxes_on_face =
          data_on_slice(neighbor_div_fluxes, neighbor_mesh.extents(),
                        dimension_in_neighbor, neighbor_face_slice_index);
      auto neighbor_normal_dot_fluxes = normal_dot_flux<all_fields_tags>(
          neighbor_face_normal_and_magnitude.first, neighbor_fluxes_on_face);
      auto neighbor_fluxes_args_on_face = std::apply(
          [&mortar_id, &neighbor_mesh, direction_from_neighbor](
              const auto&... expanded_overlap_fluxes_args) noexcept {
            return std::make_tuple(SubdomainOperator_detail::slice_arg_to_face(
                SubdomainOperator_detail::unmap_overlap_arg(
                    expanded_overlap_fluxes_args, mortar_id),
                neighbor_mesh.extents(), direction_from_neighbor)...);
          },
          all_overlap_fluxes_args);
      remote_boundary_data =
          SubdomainOperator_detail::package_boundary_data<BoundaryData>(
              numerical_fluxes_computer, fluxes_computer, neighbor_face_mesh,
              neighbor_face_normal_and_magnitude.first,
              neighbor_normal_dot_fluxes, neighbor_div_fluxes_on_face,
              neighbor_fluxes_args_on_face, AuxiliaryFields{});
      // TODO: does the mortar mesh and size need orientation?
      // TODO: is the mortar mesh and size correct for hp-refinement?
      // It's probably easiest to just communicate all mortar meshes and sizes,
      // since we need the perpendicular ones anyway.
      if (::dg::needs_projection(neighbor_face_mesh, mortar_mesh,
                                 mortar_size)) {
        remote_boundary_data = remote_boundary_data.project_to_mortar(
            neighbor_face_mesh, mortar_mesh, mortar_size);
      }

      // Apply the boundary contribution to the neighbor overlap
      // We need to orient the mortar data from the central element to the
      // neighbor since it crosses an element boundary here.
      auto reoriented_local_boundary_data = local_boundary_data;
      if (not orientation.is_aligned()) {
        reoriented_local_boundary_data.orient_on_slice(mortar_mesh.extents(),
                                                       dimension, orientation);
      }
      SubdomainOperator_detail::apply_boundary_contribution(
          make_not_null(&neighbor_result_extended), numerical_fluxes_computer,
          remote_boundary_data, std::move(reoriented_local_boundary_data),
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
           all_overlap_mortar_meshes.at(mortar_id)) {
        const auto& neighbor_mortar_id = neighbor_mortar_id_and_mesh.first;
        const auto& neighbor_face_direction = neighbor_mortar_id.first;
        if (neighbor_face_direction.dimension() ==
            direction_from_neighbor.dimension()) {
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
            all_overlap_mortar_sizes.at(mortar_id).at(neighbor_mortar_id);

        neighbor_face_normal_and_magnitude =
            SubdomainOperator_detail::face_normal_and_magnitude(
                neighbor_face_mesh, all_overlap_element_maps.at(mortar_id),
                neighbor_face_direction);

        neighbor_fluxes_on_face =
            data_on_slice(neighbor_fluxes, neighbor_mesh.extents(),
                          neighbor_face_dimension, neighbor_face_slice_index);
        neighbor_div_fluxes_on_face =
            data_on_slice(neighbor_div_fluxes, neighbor_mesh.extents(),
                          neighbor_face_dimension, neighbor_face_slice_index);
        neighbor_normal_dot_fluxes = normal_dot_flux<all_fields_tags>(
            neighbor_face_normal_and_magnitude.first, neighbor_fluxes_on_face);
        neighbor_fluxes_args_on_face = std::apply(
            [&neighbor_mesh, &mortar_id, &neighbor_face_direction](
                const auto&... expanded_overlap_fluxes_args) noexcept {
              return std::make_tuple(
                  SubdomainOperator_detail::slice_arg_to_face(
                      SubdomainOperator_detail::unmap_overlap_arg(
                          expanded_overlap_fluxes_args, mortar_id),
                      neighbor_mesh.extents(), neighbor_face_direction)...);
            },
            all_overlap_fluxes_args);
        auto neighbor_local_boundary_data =
            SubdomainOperator_detail::package_boundary_data<BoundaryData>(
                numerical_fluxes_computer, fluxes_computer, neighbor_face_mesh,
                neighbor_face_normal_and_magnitude.first,
                neighbor_normal_dot_fluxes, neighbor_div_fluxes_on_face,
                neighbor_fluxes_args_on_face, AuxiliaryFields{});
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
              numerical_fluxes_computer, neighbor_fluxes_args_on_face);
        } else {
          // Assume the data on the neighbor's neighbor is zero.

          // TODO: Make sure this works with h-refinement.. we know the
          // data on mortars to other neighbors in the same direction, so it
          // shouldn't be zero.

          // The normal is probably irrelevant in this case, but we compute it
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
                  db::item_type<n_dot_fluxes_tag>{neighbor_face_num_points, 0.},
                  db::item_type<div_fluxes_tag>{neighbor_face_num_points, 0.},
                  // TODO: make sure using these args is fine
                  neighbor_fluxes_args_on_face, AuxiliaryFields{});
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
      const auto neighbor_result = LinearSolver::Schwarz::data_on_overlap(
          neighbor_result_extended, neighbor_mesh.extents(),
          all_overlap_extents.at(mortar_id), direction_from_neighbor);
      // TODO: Fake boundary contributions from the other mortars of the
      // neighbor by filling their data with zeros
      //   Parallel::printf("Final result on overlap: %s\n", neighbor_result);

      result->overlap_data.insert_or_assign(mortar_id,
                                            std::move(neighbor_result));

      // Prepare applying the boundary contribution to the central element
      // We need to orient the mortar data from the neighbor to the central
      // element since it crosses an element boundary here.
      if (not orientation.is_aligned()) {
        remote_boundary_data.orient_on_slice(mortar_mesh.extents(),
                                             dimension_in_neighbor,
                                             orientation.inverse_map());
      }
    }

    // Apply the boundary contribution to the central element
    SubdomainOperator_detail::apply_boundary_contribution(
        make_not_null(&result->element_data), numerical_fluxes_computer,
        local_boundary_data, std::move(remote_boundary_data),
        magnitude_of_face_normal, mesh, direction, mortar_mesh, mortar_size);
  }
}

template <typename Tag, typename Dim, typename OptionsGroup,
          typename CenterTags>
struct make_overlap_tag_impl {
  using type = tmpl::conditional_t<
      tmpl::list_contains_v<CenterTags, Tag>, Tag,
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim::value, OptionsGroup>>;
};

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename FluxesComputerTag, typename FluxesArgs,
          typename SourcesComputer, typename SourcesArgs,
          typename NumericalFluxesComputerTag, typename OptionsGroup,
          typename FluxesArgsTagsFromCenter>
struct SubdomainOperator {
 private:
  using all_fields_tags = tmpl::append<PrimalFields, AuxiliaryFields>;
  using vars_tag = ::Tags::Variables<all_fields_tags>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using FluxesComputerType = db::const_item_type<FluxesComputerTag>;
  using NumericalFluxesComputerType =
      db::const_item_type<NumericalFluxesComputerTag>;

  using fluxes_args_tags = typename FluxesComputerType::argument_tags;
  static constexpr size_t num_fluxes_args = tmpl::size<fluxes_args_tags>::value;
  using sources_args_tags = typename SourcesComputer::argument_tags;
  static constexpr size_t num_sources_args =
      tmpl::size<sources_args_tags>::value;

  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

  using Buffer = tuples::TaggedTuple<fluxes_tag, div_fluxes_tag>;

 public:
  static constexpr size_t volume_dim = Dim;

  explicit SubdomainOperator(const size_t element_num_points) noexcept
      : buffer_{db::item_type<fluxes_tag>{element_num_points},
                db::item_type<div_fluxes_tag>{element_num_points}} {}

  struct element_operator {
    using argument_tags =
        tmpl::append<tmpl::list<domain::Tags::Mesh<Dim>,
                                domain::Tags::InverseJacobian<
                                    Dim, Frame::Logical, Frame::Inertial>,
                                FluxesComputerTag>,
                     fluxes_args_tags, sources_args_tags>;

    template <typename... RemainingArgs>
    static void apply(
        const Mesh<Dim>& mesh,
        const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
            inv_jacobian,
        const FluxesComputerType& fluxes_computer,
        const RemainingArgs&... expanded_remaining_args) noexcept {
      const std::tuple<RemainingArgs...> remaining_args{
          expanded_remaining_args...};
      const auto& arg = get<sizeof...(RemainingArgs) - 3>(remaining_args);
      const auto& result = get<sizeof...(RemainingArgs) - 2>(remaining_args);
      const auto& subdomain_operator =
          get<sizeof...(RemainingArgs) - 1>(remaining_args);
      apply_subdomain_center_volume<PrimalFields, AuxiliaryFields,
                                    SourcesComputer>(
          make_not_null(&(result->element_data)),
          make_not_null(&get<fluxes_tag>(subdomain_operator->buffer_)),
          make_not_null(&get<div_fluxes_tag>(subdomain_operator->buffer_)),
          fluxes_computer, mesh, inv_jacobian,
          tuple_head<num_fluxes_args>(remaining_args),
          tuple_slice<num_fluxes_args, num_fluxes_args + num_sources_args>(
              remaining_args),
          arg.element_data);
    }
  };

  template <typename Directions>
  struct face_operator {
    using overlap_fluxes_args_tags = tmpl::transform<
        fluxes_args_tags,
        make_overlap_tag_impl<tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>,
                              tmpl::pin<OptionsGroup>,
                              tmpl::pin<FluxesArgsTagsFromCenter>>>;
    using argument_tags = tmpl::append<
        tmpl::list<
            domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
            FluxesComputerTag, NumericalFluxesComputerTag,
            domain::Tags::Direction<Dim>,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>,
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
            overlaps_tag<domain::Tags::Extents<Dim>>,
            overlaps_tag<domain::Tags::Mesh<Dim>>,
            overlaps_tag<domain::Tags::ElementMap<Dim>>,
            overlaps_tag<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
            overlaps_tag<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>>,
        fluxes_args_tags, overlap_fluxes_args_tags,
        db::wrap_tags_in<overlaps_tag, sources_args_tags>>;
    using volume_tags = tmpl::append<
        tmpl::list<
            domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
            FluxesComputerTag, NumericalFluxesComputerTag,
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
            overlaps_tag<domain::Tags::Extents<Dim>>,
            overlaps_tag<domain::Tags::Mesh<Dim>>,
            overlaps_tag<domain::Tags::ElementMap<Dim>>,
            overlaps_tag<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
            overlaps_tag<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>>,
        get_volume_tags<FluxesComputerType>, overlap_fluxes_args_tags,
        db::wrap_tags_in<overlaps_tag, sources_args_tags>>;

    // interface_apply doesn't currently support `void` return types
    template <typename... RemainingArgs>
    int operator()(
        const Element<Dim>& element, const Mesh<Dim>& mesh,
        const FluxesComputerType& fluxes_computer,
        const NumericalFluxesComputerType& numerical_fluxes_computer,
        const Direction<Dim>& direction,
        const tnsr::i<DataVector, Dim>& face_normal,
        const Scalar<DataVector>& face_normal_magnitude,
        const db::const_item_type<
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>& mortar_meshes,
        const db::const_item_type<
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
        const overlaps<Index<Dim>>& all_overlap_extents,
        const overlaps<Mesh<Dim>>& all_overlap_meshes,
        const overlaps<ElementMap<Dim, Frame::Inertial>>&
            all_overlap_element_maps,
        const overlaps<::dg::MortarMap<Dim, Mesh<Dim - 1>>>&
            all_overlap_mortar_meshes,
        const overlaps<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>&
            all_overlap_mortar_sizes,
        RemainingArgs&&... expanded_remaining_args) const noexcept {
      const auto remaining_args =
          std::forward_as_tuple(expanded_remaining_args...);
      const auto& arg = get<sizeof...(RemainingArgs) - 3>(remaining_args);
      const auto& result = get<sizeof...(RemainingArgs) - 2>(remaining_args);
      const auto& subdomain_operator =
          get<sizeof...(RemainingArgs) - 1>(remaining_args);
      apply_subdomain_face<PrimalFields, AuxiliaryFields, SourcesComputer>(
          result, element, mesh, fluxes_computer, numerical_fluxes_computer,
          direction, face_normal, face_normal_magnitude, mortar_meshes,
          mortar_sizes, all_overlap_extents, all_overlap_meshes,
          all_overlap_element_maps, all_overlap_mortar_meshes,
          all_overlap_mortar_sizes, tuple_head<num_fluxes_args>(remaining_args),
          tuple_slice<num_fluxes_args, 2 * num_fluxes_args>(remaining_args),
          tuple_slice<2 * num_fluxes_args,
                      2 * num_fluxes_args + num_sources_args>(remaining_args),
          arg, get<fluxes_tag>(subdomain_operator->buffer_),
          get<div_fluxes_tag>(subdomain_operator->buffer_));
      return 0;
    }
  };

 private:
  Buffer buffer_;
};

}  // namespace dg
}  // namespace elliptic
