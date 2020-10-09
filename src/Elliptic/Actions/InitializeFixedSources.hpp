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
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

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
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Retrieve the sources of the elliptic system from the analytic solution,
    // which defines the problem we want to solve.
    // We need only retrieve sources for the primal fields, since the auxiliary
    // fields will never be sourced.
    typename fixed_sources_tag::type fixed_sources{num_grid_points, 0.};
    fixed_sources.assign_subset(
        Parallel::get<typename Metavariables::background_tag>(cache)
            .variables(inertial_coords,
                       db::wrap_tags_in<::Tags::FixedSource,
                                        typename system::primal_fields>{}));
    if constexpr (Metavariables::massive_operator) {
      fixed_sources = mass(
          fixed_sources, mesh,
          db::get<domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>>(
              box));
    }

    const auto& boundary_conditions_provider =
        Parallel::get<typename Metavariables::boundary_conditions_tag>(cache);
    const auto& linearized_boundary_conditions_provider = Parallel::get<
        typename Metavariables::linearized_boundary_conditions_tag>(cache);

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
      tmpl::for_each<typename system::primal_fields>([&](auto tag_v) noexcept {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<elliptic::Tags::BoundaryCondition<tag>>(
            boundary_conditions[direction]) =
            boundary_conditions_provider.boundary_condition_type(
                db::get<domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsExterior<Dim>,
                    domain::Tags::Coordinates<Dim, Frame::Inertial>>>(box)
                    .at(direction),
                direction, tag{});
        using lin_tag =
            tmpl::at<typename Metavariables::primal_variables,
                     tmpl::index_of<typename system::primal_fields, tag>>;
        get<elliptic::Tags::BoundaryCondition<lin_tag>>(
            lin_boundary_conditions[direction]) =
            linearized_boundary_conditions_provider.boundary_condition_type(
                db::get<domain::Tags::Interface<
                    domain::Tags::BoundaryDirectionsExterior<Dim>,
                    domain::Tags::Coordinates<Dim, Frame::Inertial>>>(box)
                    .at(direction),
                direction, tag{});
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

      typename background_fields_tag::type background_fields_vars{
          num_grid_points, 0.};
      background_fields_vars.assign_subset(
          Parallel::get<typename Metavariables::background_tag>(cache)
              .variables(inertial_coords, background_fields{}));

      return std::make_tuple(
          ::Initialization::merge_into_databox<
              InitializeFixedSources, db::AddSimpleTags<background_fields_tag>,
              db::AddComputeTags<domain::Tags::Slice<
                  domain::Tags::BoundaryDirectionsInterior<Dim>,
                  background_fields_tag>>>(std::move(new_box),
                                           std::move(background_fields_vars)));
    }
  }
};

}  // namespace Actions
}  // namespace elliptic
