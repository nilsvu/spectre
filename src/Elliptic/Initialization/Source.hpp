// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Computes the sources of the elliptic equations and adds them to the
 * DataBox
 *
 * \note Currently the sources are always retrieved from an analytic solution.
 *
 * With:
 * - `sources_tag` = `db::add_tag_prefix<Tags::Source, system::fields_tag>`
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 * - ConstGlobalCache:
 *   - `analytic_solution_tag`
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Coordinates<volume_dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `sources_tag`
 */
template <typename Metavariables, typename = cpp17::void_t<>>
struct Source {
  using system = typename Metavariables::system;

  using sources_tag =
      db::add_tag_prefix<::Tags::Source, typename system::fields_tag>;

  using simple_tags = db::AddSimpleTags<sources_tag>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    const auto& inertial_coords =
        get<::Tags::Coordinates<system::volume_dim, Frame::Inertial>>(box);
    const auto num_grid_points =
        get<::Tags::Mesh<system::volume_dim>>(box).number_of_grid_points();

    db::item_type<sources_tag> sources(num_grid_points, 0.);
    // This actually sets the complete set of tags in the Variables, but there
    // is no Variables constructor from a TaggedTuple (yet)
    sources.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords,
                       db::get_variables_tags_list<sources_tag>{}));

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(sources));
  }
};

template <typename Metavariables>
struct Source<Metavariables,
              cpp17::void_t<typename Metavariables::nonlinear_solver>> {
  using system = typename Metavariables::system;

  using nonlinear_sources_tag =
      db::add_tag_prefix<::Tags::Source, typename system::nonlinear_fields_tag>;

  using background_tags = typename system::background_tags;
  using background_vars_tag = ::Tags::Variables<background_tags>;

  using simple_tags =
      db::AddSimpleTags<nonlinear_sources_tag, background_vars_tag>;
//   using simple_tags =
//       db::AddSimpleTags<nonlinear_sources_tag>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    const auto& inertial_coords =
        get<::Tags::Coordinates<system::volume_dim, Frame::Inertial>>(box);
    const auto num_grid_points =
        get<::Tags::Mesh<system::volume_dim>>(box).number_of_grid_points();

    db::item_type<nonlinear_sources_tag> nonlinear_sources(num_grid_points, 0.);
    // This actually sets the complete set of tags in the Variables, but there
    // is no Variables constructor from a TaggedTuple (yet)
    nonlinear_sources.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords,
                       db::get_variables_tags_list<nonlinear_sources_tag>{}));
    db::item_type<background_vars_tag> background_vars(num_grid_points, 0.);
    background_vars.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords, background_tags{}));

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(nonlinear_sources),
        std::move(background_vars));
    // return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
    //     std::move(box), std::move(nonlinear_sources));
  }
};
}  // namespace Initialization
}  // namespace Elliptic
