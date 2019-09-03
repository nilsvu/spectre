// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace FirstOrderEllipticSolutionsTestHelpers {

namespace Tags {

template <typename Tag>
struct OperatorAppliedTo : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  static std::string name() noexcept { return "OperatorAppliedTo"; }
  using tag = Tag;
};

template <size_t Dim>
struct InverseJacobian : db::SimpleTag {
  using type =
      ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>;
  static std::string name() noexcept { return "InverseJacobian"; }
};

}  // namespace Tags

/// Tests that the `solution` numerically solves the `System` on the given grid
template <typename System, typename SolutionType,
          size_t Dim = System::volume_dim, typename... Maps>
void verify_solution(
    const SolutionType& solution,
    const typename System::fluxes& fluxes_computer, const Mesh<Dim>& mesh,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>
        coord_map,
    const double tolerance) {
  constexpr size_t volume_dim = System::volume_dim;
  using fields_tag = typename System::fields_tag;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, fields_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<Tags::OperatorAppliedTo, fields_tag>;
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;

  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = coord_map(logical_coords);
  const auto solution_fields = variables_from_tagged_tuple(solution.variables(
      inertial_coords, db::get_variables_tags_list<fields_tag>{}));

  auto box = db::create<
      db::AddSimpleTags<::Tags::Mesh<volume_dim>, Tags::InverseJacobian<Dim>,
                        elliptic::Tags::FluxesComputer<typename System::fluxes>,
                        fields_tag, operator_applied_to_fields_tag>,
      db::AddComputeTags<
          elliptic::Tags::FirstOrderFluxesCompute<
              volume_dim, System, typename System::fields_tag,
              typename System::fluxes, typename System::primal_fields,
              typename System::auxiliary_fields>,
          ::Tags::DivCompute<fluxes_tag, Tags::InverseJacobian<Dim>>,
          elliptic::Tags::FirstOrderSourcesCompute<
              System, typename System::fields_tag, typename System::sources,
              typename System::primal_fields,
              typename System::auxiliary_fields>>>(
      mesh, coord_map.inv_jacobian(logical_coords), fluxes_computer,
      solution_fields,
      make_with_value<db::item_type<operator_applied_to_fields_tag>>(
          inertial_coords, std::numeric_limits<double>::signaling_NaN()));

  db::mutate_apply<elliptic::FirstOrderOperator<
      volume_dim, Tags::OperatorAppliedTo, fields_tag>>(make_not_null(&box));

  auto fixed_sources =
      make_with_value<db::item_type<fixed_sources_tag>>(inertial_coords, 0.);
  fixed_sources.assign_subset(solution.variables(
      inertial_coords,
      db::wrap_tags_in<::Tags::FixedSource, typename System::primal_fields>{}));

  Approx numerical_approx = Approx::custom().epsilon(tolerance).scale(1.);
  CHECK_VARIABLES_CUSTOM_APPROX(
      get<operator_applied_to_fields_tag>(box),
      db::item_type<operator_applied_to_fields_tag>(fixed_sources),
      numerical_approx);
}

}  // namespace FirstOrderEllipticSolutionsTestHelpers
