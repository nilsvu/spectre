// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

namespace detail {
template <typename BackgroundFields, size_t Dim, typename Background>
Variables<BackgroundFields> all_background_fields(
    const Background& background,
    const tnsr::I<DataVector, Dim>& inertial_coords, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inv_jacobian) noexcept {
  using background_fields = BackgroundFields;

  using derived_background_fields =
      tmpl::list<Xcts::Tags::ConformalChristoffelFirstKind<DataVector, Dim,
                                                           Frame::Inertial>,
                 Xcts::Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                            Frame::Inertial>,
                 Xcts::Tags::ConformalChristoffelContracted<DataVector, Dim,
                                                            Frame::Inertial>,
                 Xcts::Tags::ConformalRicciScalar<DataVector>,
                 ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                               tmpl::size_t<Dim>, Frame::Inertial>>;
  using queried_background_fields =
      tmpl::list_difference<background_fields, derived_background_fields>;

  const size_t num_grid_points = inertial_coords.begin()->size();
  Variables<background_fields> background_fields_vars{num_grid_points};
  background_fields_vars.assign_subset(
      background.variables(inertial_coords, queried_background_fields{}));

  if constexpr (tmpl::list_contains_v<
                    background_fields,
                    Xcts::Tags::ConformalRicciScalar<DataVector>>) {
    const auto& inv_conformal_metric = get<
        Xcts::Tags::InverseConformalMetric<DataVector, Dim, Frame::Inertial>>(
        background_fields_vars);
    const auto deriv_conformal_metric = get<::Tags::deriv<
        Xcts::Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>,
        tmpl::size_t<Dim>, Frame::Inertial>>(
        background.variables(
            inertial_coords,
            tmpl::list<::Tags::deriv<
                Xcts::Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>,
                tmpl::size_t<Dim>, Frame::Inertial>>{}));
    gr::christoffel_first_kind(
        make_not_null(
            &get<Xcts::Tags::ConformalChristoffelFirstKind<
                DataVector, Dim, Frame::Inertial>>(background_fields_vars)),
        deriv_conformal_metric);
    const auto& conformal_christoffel_first_kind =
        get<Xcts::Tags::ConformalChristoffelFirstKind<DataVector, Dim,
                                                      Frame::Inertial>>(
            background_fields_vars);
    raise_or_lower_first_index(
        make_not_null(
            &get<Xcts::Tags::ConformalChristoffelSecondKind<
                DataVector, Dim, Frame::Inertial>>(background_fields_vars)),
        conformal_christoffel_first_kind, inv_conformal_metric);
    const auto& conformal_christoffel_second_kind =
        get<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                       Frame::Inertial>>(
            background_fields_vars);
    for (size_t i = 0; i < Dim; ++i) {
      get<Xcts::Tags::ConformalChristoffelContracted<DataVector, Dim,
                                                     Frame::Inertial>>(
          background_fields_vars)
          .get(i) = 0.;
      for (size_t j = 0; j < Dim; ++j) {
        get<Xcts::Tags::ConformalChristoffelContracted<DataVector, Dim,
                                                       Frame::Inertial>>(
            background_fields_vars)
            .get(i) += conformal_christoffel_second_kind.get(j, i, j);
      }
    }
    Variables<tmpl::list<Xcts::Tags::ConformalChristoffelSecondKind<
                             DataVector, Dim, Frame::Inertial>,
                         gr::Tags::TraceExtrinsicCurvature<DataVector>>>
        background_fields_to_derive{num_grid_points};
    get<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                   Frame::Inertial>>(
        background_fields_to_derive) = conformal_christoffel_second_kind;
    get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
        background_fields_to_derive) =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
            background_fields_vars);
    const auto deriv_background_fields = partial_derivatives<
        tmpl::list<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                              Frame::Inertial>,
                   gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
        background_fields_to_derive, mesh, inv_jacobian);
    const auto& deriv_conformal_christoffel_second_kind =
        get<::Tags::deriv<Xcts::Tags::ConformalChristoffelSecondKind<
                              DataVector, Dim, Frame::Inertial>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
            deriv_background_fields);
    const auto conformal_ricci_tensor =
        gr::ricci_tensor(conformal_christoffel_second_kind,
                         deriv_conformal_christoffel_second_kind);
    trace(make_not_null(&get<Xcts::Tags::ConformalRicciScalar<DataVector>>(
              background_fields_vars)),
          conformal_ricci_tensor, inv_conformal_metric);
    get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                      tmpl::size_t<Dim>, Frame::Inertial>>(
        background_fields_vars) =
        get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
            deriv_background_fields);
  } else {
    Variables<tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>>>
        background_fields_to_derive{num_grid_points};
    get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
        background_fields_to_derive) =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
            background_fields_vars);
    const auto deriv_background_fields = partial_derivatives<
        tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
        background_fields_to_derive, mesh, inv_jacobian);
    get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                      tmpl::size_t<Dim>, Frame::Inertial>>(
        background_fields_vars) =
        get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
            deriv_background_fields);
  }
  return background_fields_vars;
}
}

struct InitializeFixedSources {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using fixed_sources_tag =
        db::add_tag_prefix<::Tags::FixedSource, fields_tag>;

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& element_map = db::get<domain::Tags::ElementMap<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& background =
        Parallel::get<typename Metavariables::background_tag>(cache);

    // Retrieve the sources of the elliptic system from the background, which
    // defines the problem we want to solve (along with the boundary
    // conditions). We need only retrieve sources for the primal fields, since
    // the auxiliary fields will never be sourced.
    typename fixed_sources_tag::type fixed_sources{num_grid_points, 0.};
    fixed_sources.assign_subset(background.variables(
        inertial_coords, db::wrap_tags_in<::Tags::FixedSource,
                                          typename system::primal_fields>{}));
    if constexpr (Metavariables::massive_operator) {
      fixed_sources = mass(
          fixed_sources, mesh,
          db::get<domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>>(
              box));
    }

    const auto& boundary_conditions_provider =
        Parallel::get<typename Metavariables::boundary_conditions_tag>(cache);

    std::unordered_map<
        Direction<Dim>,
        tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
            elliptic::Tags::BoundaryCondition, typename system::primal_fields>>>
        boundary_conditions{};
    std::unordered_map<Direction<Dim>,
                       tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
                           elliptic::Tags::BoundaryCondition,
                           typename Metavariables::primal_variables>>>
        lin_boundary_conditions{};
    for (const auto& direction :
         db::get<domain::Tags::Element<Dim>>(box).external_boundaries()) {
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const auto face_inertial_coords =
          element_map(interface_logical_coordinates(face_mesh, direction));
      tmpl::for_each<typename system::primal_fields>([&](auto tag_v) noexcept {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<elliptic::Tags::BoundaryCondition<tag>>(
            boundary_conditions[direction]) =
            boundary_conditions_provider.boundary_condition_type(
                face_inertial_coords, direction, tag{});
        using lin_tag =
            tmpl::at<typename Metavariables::primal_variables,
                     tmpl::index_of<typename system::primal_fields, tag>>;
        get<elliptic::Tags::BoundaryCondition<lin_tag>>(
            lin_boundary_conditions[direction]) =
            get<elliptic::Tags::BoundaryCondition<tag>>(
                boundary_conditions[direction]);
      });
    }

    auto new_box = ::Initialization::merge_into_databox<
        InitializeFixedSources,
        db::AddSimpleTags<fixed_sources_tag,
                          domain::Tags::Interface<
                              domain::Tags::BoundaryDirectionsExterior<Dim>,
                              elliptic::Tags::BoundaryConditions<
                                  typename system::primal_fields>>,
                          domain::Tags::Interface<
                              domain::Tags::BoundaryDirectionsExterior<Dim>,
                              elliptic::Tags::BoundaryConditions<
                                  typename Metavariables::primal_variables>>>>(
        std::move(box), std::move(fixed_sources),
        std::move(boundary_conditions), std::move(lin_boundary_conditions));

    if constexpr (tmpl::size<typename system::background_fields>::value == 0) {
      return std::make_tuple(std::move(new_box));
    } else {
      using background_fields = typename system::background_fields;
      using background_fields_tag = ::Tags::Variables<background_fields>;

      auto background_fields_vars =
          detail::all_background_fields<background_fields>(
              background, inertial_coords, mesh,
              get<domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                Frame::Inertial>>(new_box));
      return std::make_tuple(
          ::Initialization::merge_into_databox<
              InitializeFixedSources, db::AddSimpleTags<background_fields_tag>>(
              std::move(new_box), std::move(background_fields_vars)));
    }
  }
};

}  // namespace Actions
}  // namespace elliptic
