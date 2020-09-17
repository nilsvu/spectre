// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

/*!
 * \brief Initialize those DataBox tags related to the elliptic system that are
 * independent of the dynamic variables
 *
 * This action initializes the "fixed sources", i.e. the variable-independent
 * source-term \f$f(x)\f$ in an elliptic system of PDEs \f$-div(F) + S =
 * f(x)\f$.
 *
 * Uses:
 * - Metavariables:
 *   - `background_tag`
 * - System:
 *   - `fields_tag`
 *   - `primal_fields`
 * - DataBox:
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `db::add_tag_prefix<::Tags::FixedSource, fields_tag>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename System, typename BackgroundTag>
struct InitializeFixedSources {
 private:
  using system = System;
  using fields_tag = typename system::fields_tag;
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;

 public:
  using simple_tags = tmpl::list<fixed_sources_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& background = db::get<BackgroundTag>(box);

    // Retrieve the fixed-sources of the elliptic system from the background,
    // which (along with the boundary conditions) define the problem we want to
    // solve. We need only retrieve sources for the primal fields, since the
    // auxiliary fields will never be sourced.
    auto fixed_sources =
        make_with_value<typename fixed_sources_tag::type>(inertial_coords, 0.);
    fixed_sources.assign_subset(background.variables(
        inertial_coords, db::wrap_tags_in<::Tags::FixedSource,
                                          typename system::primal_fields>{}));

    if constexpr (Metavariables::massive_operator) {
      fixed_sources = mass(
          fixed_sources, db::get<domain::Tags::Mesh<Dim>>(box),
          db::get<domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>>(
              box));
    }

    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(fixed_sources));
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
