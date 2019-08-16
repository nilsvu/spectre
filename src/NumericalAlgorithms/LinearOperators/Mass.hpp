// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions and tags for taking a divergence.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;

namespace Tags {
template <size_t Dim>
struct Mesh;
/// \endcond

template <typename Tag>
struct Mass : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "Mass(" + Tag::name() + ")"; }
  using tag = Tag;
  using type = typename Tag::type;
};

}  // namespace Tags

template <typename VariablesTags, size_t Dim, typename IntegrationFrame>
Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>> mass(
    const Variables<VariablesTags>& variables, const Mesh<Dim>& mesh,
    const Jacobian<DataVector, Dim, Frame::Logical, IntegrationFrame>&
        jacobian) noexcept {
  std::array<Matrix, Dim> mass_matrices{};
  DataVector jacobian_determinant{mesh.number_of_grid_points(), 1.};
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(mass_matrices, d) = Spectral::mass_matrix(mesh.slice_through(d));
    // Jacobian is diagonal, so we just multiply diagonal elements for the
    // determinant.
    jacobian_determinant *= jacobian.get(d, d);
  }
  return Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>>(apply_matrices(
      mass_matrices, variables * jacobian_determinant, mesh.extents()));
}

// namespace Tags {
// template <typename VariablesTag, typename InverseJacobianTag>
// struct MassCompute : db::add_tag_prefix<Mass, VariablesTag>, db::ComputeTag {
//  private:
//   using inv_jac_indices =
//       typename db::item_type<InverseJacobianTag>::index_list;
//   static constexpr size_t dim = tmpl::back<inv_jac_indices>::dim;

//  public:
//   using argument_tags =
//       tmpl::list<VariablesTag, Tags::Mesh<dim>, InverseJacobianTag>;
//   static constexpr auto function =
//       mass<db::get_variables_tags_list<db::item_type<VariablesTag>>, dim,
//            typename tmpl::back<inv_jac_indices>::Frame>;
// };
// }  // namespace Tags

namespace Mass_detail {

template <size_t VolumeDim>
struct MassOnFaceImpl {
  template <typename VariablesTags, typename IntegrationFrame>
  static Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>> apply(
      const Variables<VariablesTags>& variables_on_face,
      const Mesh<VolumeDim - 1>& face_mesh, const size_t face_dimension,
      const Jacobian<DataVector, VolumeDim, Frame::Logical, IntegrationFrame>&
          volume_jacobian_on_face) noexcept {
    std::array<Matrix, VolumeDim - 1> mass_matrices{};
    DataVector jacobian_determinant{face_mesh.number_of_grid_points(), 1.};
    for (size_t d = 0; d < VolumeDim - 1; d++) {
      gsl::at(mass_matrices, d) =
          Spectral::mass_matrix(face_mesh.slice_through(d));
      // Jacobian is diagonal, so we just multiply diagonal elements for the
      // determinant. We omit the dimension that the face slices through to get
      // the surface jacobian determinant.
      const size_t d_in_volume = d + (d >= face_dimension ? 1 : 0);
      jacobian_determinant *=
          volume_jacobian_on_face.get(d_in_volume, d_in_volume);
    }
    return Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>>(
        apply_matrices(mass_matrices, variables_on_face * jacobian_determinant,
                       face_mesh.extents()));
  }
};

template <>
struct MassOnFaceImpl<1> {
  template <typename VariablesTags, typename IntegrationFrame>
  static Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>> apply(
      const Variables<VariablesTags>& variables_on_face,
      const Mesh<0>& /*face_mesh*/, const size_t /*face_dimension*/,
      const Jacobian<DataVector, 1, Frame::Logical, IntegrationFrame>&
      /*volume_jacobian_on_face*/) noexcept {
    return Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>>(
        variables_on_face);
  }
};
}  // namespace Mass_detail

template <typename VariablesTags, size_t VolumeDim, typename IntegrationFrame>
Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>> mass_on_face(
    const Variables<VariablesTags>& variables_on_face,
    const Mesh<VolumeDim - 1>& face_mesh, const size_t face_dimension,
    const Jacobian<DataVector, VolumeDim, Frame::Logical, IntegrationFrame>&
        volume_jacobian_on_face) noexcept {
  return Mass_detail::MassOnFaceImpl<VolumeDim>::apply(
      variables_on_face, face_mesh, face_dimension, volume_jacobian_on_face);
}

template <typename VariablesTags, size_t VolumeDim, typename DiffFrame>
Variables<db::wrap_tags_in<::Tags::div, VariablesTags>> stiffness(
    const Variables<VariablesTags>& variables, const Mesh<VolumeDim>& mesh,
    const InverseJacobian<DataVector, VolumeDim, Frame::Logical, DiffFrame>&
        inv_jacobian) noexcept {
  Variables<db::wrap_tags_in<::Tags::div, VariablesTags>> result{
      variables.number_of_grid_points(), 0.};
  for (size_t d = 0; d < VolumeDim; d++) {
    std::array<Matrix, VolumeDim> diff_matrices_transpose{};
    Matrix& diff_matrix = gsl::at(diff_matrices_transpose, d);
    diff_matrix =
        trans(Spectral::differentiation_matrix(mesh.slice_through(d)));
    // // Jacobian is diagonal
    // const DataVector& jacobian_factor = inv_jacobian.get(d, d);
    // for (size_t i = 0; i < jacobian_factor.size(); i++) {
    //   for (size_t j = 0; j < jacobian_factor.size(); j++) {
    //     diff_matrix(i, j) *= jacobian_factor[j];
    //   }
    // }
    // We only need the d-th first-index component of the tensors in
    // `variables`, so instead of taking that in the loop below we could take it
    // here already, could speed this up
    auto derivs_this_dim =
        apply_matrices(diff_matrices_transpose, variables, mesh.extents());

    tmpl::for_each<VariablesTags>([
      &result, &derivs_this_dim, &d, &inv_jacobian
    ](auto deriv_tag_v) noexcept {
      using deriv_tag = tmpl::type_from<decltype(deriv_tag_v)>;
      using div_tag = ::Tags::div<deriv_tag>;

      using first_index =
          tmpl::front<typename db::item_type<deriv_tag>::index_list>;
      static_assert(
          cpp17::is_same_v<typename first_index::Frame, DiffFrame> and
              first_index::ul == UpLo::Up,
          "First index of tensor cannot be contracted with derivative "
          "because either it is in the wrong frame or it has the wrong "
          "valence");

      auto& div = get<div_tag>(result);
      for (auto it = div.begin(); it != div.end(); ++it) {
        const auto div_indices = div.get_tensor_index(it);
        const auto flux_indices = prepend(div_indices, d);
        // Jacobian is diagonal
        *it += inv_jacobian.get(d, d) *
               get<deriv_tag>(derivs_this_dim).get(flux_indices);
      }
    });
  }
  return result;
}
