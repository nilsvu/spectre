// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/DgElementArray.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace elliptic::dg::py_bindings {

/// Prefix tag that represents the elliptic DG operator applied to fields.
template <typename Tag>
struct DgOperatorAppliedTo : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <typename System>
struct Workspace {
 private:
  static constexpr size_t Dim = System::volume_dim;

 public:
  Variables<typename System::auxiliary_fields> auxiliary_vars;
  Variables<typename System::auxiliary_fluxes> auxiliary_fluxes;
  Variables<typename System::primal_fluxes> primal_fluxes;
  ::dg::MortarMap<
      Dim, ::elliptic::dg::MortarData<size_t, typename System::primal_fields,
                                      typename System::primal_fluxes>>
      all_mortar_data;
};

namespace detail {
template <typename... Tags, typename Box>
auto get_items_impl(const Box& box, tmpl::list<Tags...> /*meta*/) {
  return std::forward_as_tuple(get<Tags>(box)...);
}
}  // namespace detail

template <typename Tags, typename Box>
auto get_items(const Box& box) {
  return detail::get_items_impl(box, Tags{});
}

template <typename System, bool Linearized, typename OperatorAppliedToOperand,
          typename Operand, size_t Dim = System::volume_dim>
void apply_operator(
    const gsl::not_null<OperatorAppliedToOperand*> operator_applied_to_operand,
    const gsl::not_null<std::unordered_map<ElementId<Dim>, Workspace<System>>*>
        workspace,
    const DgElementArray<System, Linearized>& dg_element_array,
    const Operand& operand, const double penalty_parameter,
    const bool massive) {
  using fluxes_args_tags = typename System::fluxes_computer::argument_tags;
  using fluxes_args_volume_tags = typename System::fluxes_computer::volume_tags;
  using sources_computer = elliptic::get_sources_computer<System, Linearized>;
  using sources_args_tags = typename sources_computer::argument_tags;

  // TODO: choose boundary conditions
  static_assert(Linearized,
                "Only linearized systems are currently supported (for boundary "
                "conditions).");
  constexpr size_t temporal_id = 0;
  const elliptic::BoundaryConditions::AnalyticSolution<
      System, Dim, typename System::primal_fields,
      typename System::primal_fluxes,
      typename System::boundary_conditions_base::registrars>
      analytic_solution_boundary_condition{};
  const auto apply_boundary_condition = [&analytic_solution_boundary_condition](
                                            const Direction<Dim>& /*direction*/,
                                            const auto... fields_and_fluxes) {
    analytic_solution_boundary_condition.apply_linearized(fields_and_fluxes...);
  };
  const auto get_items2 = [](const auto&... args) {
    return std::forward_as_tuple(args...);
  };

  for (const auto& [element_id, dg_element] : dg_element_array) {
    auto& this_workspace = (*workspace)[element_id];
    const auto& vars = operand.at(element_id);
    const auto fluxes_args = get_items<fluxes_args_tags>(dg_element);
    const auto sources_args = get_items<sources_args_tags>(dg_element);
    DirectionMap<Dim, std::decay_t<decltype(fluxes_args)>>
        fluxes_args_on_faces{};
    for (const auto& direction : Direction<Dim>::all_directions()) {
      fluxes_args_on_faces.emplace(
          direction,
          elliptic::util::apply_at<
              domain::make_faces_tags<Dim, fluxes_args_tags,
                                      fluxes_args_volume_tags>,
              fluxes_args_volume_tags>(get_items2, dg_element, direction));
    }
    elliptic::dg::prepare_mortar_data<System, Linearized>(
        make_not_null(&this_workspace.auxiliary_vars),
        make_not_null(&this_workspace.auxiliary_fluxes),
        make_not_null(&this_workspace.primal_fluxes),
        make_not_null(&this_workspace.all_mortar_data), vars,
        get<domain::Tags::Element<Dim>>(dg_element),
        get<domain::Tags::Mesh<Dim>>(dg_element),
        get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                          Frame::Inertial>>(dg_element),
        get<domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>>(
            dg_element),
        get<domain::Tags::Faces<
            Dim, domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>>>(
            dg_element),
        get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(dg_element),
        get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(dg_element),
        temporal_id, apply_boundary_condition, fluxes_args, sources_args,
        fluxes_args_on_faces);
    // Copy mortar data across element boundaries, re-orienting it if needed
    for (const auto& [direction, neighbors] :
         get<domain::Tags::Element<Dim>>(dg_element).neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        auto oriented_neighbor_mortar_data =
            this_workspace.all_mortar_data.at(mortar_id).local_data(
                temporal_id);
        if (not orientation.is_aligned()) {
          oriented_neighbor_mortar_data.orient_on_slice(
              get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(dg_element)
                  .at(mortar_id)
                  .extents(),
              direction.dimension(), orientation);
        }
        auto& neighbor_workspace = (*workspace)[neighbor_id];
        neighbor_workspace
            .all_mortar_data[std::make_tuple(direction_from_neighbor,
                                             element_id)]
            .remote_insert(temporal_id,
                           std::move(oriented_neighbor_mortar_data));
      }
    }
  }
  for (const auto& [element_id, dg_element] : dg_element_array) {
    auto& this_result = (*operator_applied_to_operand)[element_id];
    auto& this_workspace = workspace->at(element_id);
    const auto& vars = operand.at(element_id);
    const auto sources_args = get_items<sources_args_tags>(dg_element);
    elliptic::dg::apply_operator<System, Linearized>(
        make_not_null(&this_result),
        make_not_null(&this_workspace.all_mortar_data), vars,
        this_workspace.primal_fluxes,
        get<domain::Tags::Element<Dim>>(dg_element),
        get<domain::Tags::Mesh<Dim>>(dg_element),
        get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                          Frame::Inertial>>(dg_element),
        get<domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                         Frame::Inertial>>(dg_element),
        get<domain::Tags::Faces<
            Dim, domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>>>(
            dg_element),
        get<domain::Tags::Faces<
            Dim, domain::Tags::DetSurfaceJacobian<Frame::ElementLogical,
                                                  Frame::Inertial>>>(
            dg_element),
        get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(dg_element),
        get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(dg_element),
        get<::Tags::Mortars<domain::Tags::DetSurfaceJacobian<
                                Frame::ElementLogical, Frame::Inertial>,
                            Dim>>(dg_element),
        penalty_parameter, massive, temporal_id, sources_args);
  }
}

}  // namespace elliptic::dg::py_bindings
