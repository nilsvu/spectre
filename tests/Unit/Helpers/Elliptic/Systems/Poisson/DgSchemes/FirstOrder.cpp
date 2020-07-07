// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Elliptic/Systems/Poisson/DgSchemes/FirstOrder.hpp"

#include <cstddef>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Tags.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::Poisson::dg {

template <size_t Dim>
OperatorVars<Dim> apply_first_order_operator(
    const ElementId<Dim>& element_id,
    const helpers::ElementArray<Dim>& dg_elements,
    const std::unordered_map<ElementId<Dim>, Vars<Dim>>& all_vars,
    const double penalty_parameter) {
  using system =
      ::Poisson::FirstOrderSystem<Dim, ::Poisson::Geometry::Euclidean>;
  const typename system::fluxes fluxes_computer{};

  // Shortcuts for tags
  using all_fields_tags =
      db::get_variables_tags_list<typename system::fields_tag>;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, all_fields_tags,
                                       tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;

  /// [boundary_scheme]
  // Choose a numerical flux
  using NumericalFlux =
      ::elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          Dim, ::elliptic::Tags::FluxesComputer<typename system::fluxes>,
          typename system::primal_fields, typename system::auxiliary_fields>;
  const NumericalFlux numerical_fluxes_computer{penalty_parameter};
  // Define the boundary scheme
  using BoundaryData = ::dg::FirstOrderScheme::BoundaryData<NumericalFlux>;
  const auto package_boundary_data =
      [&numerical_fluxes_computer, &fluxes_computer](
          const Mesh<Dim>& volume_mesh, const Direction<Dim>& direction,
          const tnsr::i<DataVector, Dim>& face_normal,
          const Scalar<DataVector>& /*face_normal_magnitude*/,
          const Variables<n_dot_fluxes_tags>& n_dot_fluxes,
          const Variables<div_fluxes_tags>& div_fluxes, const std::tuple<> &
          /*fluxes_args*/) -> BoundaryData {
    const auto face_mesh = volume_mesh.slice_away(direction.dimension());
    return ::dg::FirstOrderScheme::package_boundary_data(
        numerical_fluxes_computer, face_mesh, n_dot_fluxes,
        get<::Tags::NormalDotFlux<field_gradient_tag<Dim>>>(n_dot_fluxes),
        get<::Tags::div<::Tags::Flux<field_gradient_tag<Dim>, tmpl::size_t<Dim>,
                                     Frame::Inertial>>>(div_fluxes),
        face_normal, fluxes_computer);
  };
  const auto apply_boundary_contribution =
      [&numerical_fluxes_computer](
          const auto result, const BoundaryData& local_boundary_data,
          const BoundaryData& remote_boundary_data,
          const Scalar<DataVector>& magnitude_of_face_normal,
          const Mesh<Dim>& mesh, const ::dg::MortarId<Dim>& mortar_id,
          const Mesh<Dim - 1>& mortar_mesh,
          const ::dg::MortarSize<Dim - 1>& mortar_size) {
        const size_t dimension = mortar_id.first.dimension();
        auto boundary_contribution = std::decay_t<decltype(*result)>{
            ::dg::FirstOrderScheme::boundary_flux(
                local_boundary_data, remote_boundary_data,
                numerical_fluxes_computer, magnitude_of_face_normal,
                mesh.extents(dimension), mesh.slice_away(dimension),
                mortar_mesh, mortar_size)};
        add_slice_to_data(result, std::move(boundary_contribution),
                          mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), mortar_id.first));
      };
  /// [boundary_scheme]

  return helpers::apply_first_order_dg_operator<system>(
      element_id, dg_elements, all_vars, fluxes_computer,
      // The Poisson fluxes and sources need no arguments, so we return
      // empty tuples
      [](const auto&...) { return std::tuple<>(); },
      [](const auto&...) { return std::tuple<>(); }, package_boundary_data,
      apply_boundary_contribution);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template OperatorVars<DIM(data)> apply_first_order_operator(                \
      const ElementId<DIM(data)>&, const helpers::ElementArray<DIM(data)>&, \
      const std::unordered_map<ElementId<DIM(data)>, Vars<DIM(data)>>&,       \
      double);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace TestHelpers::Poisson::dg
