// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"

/// \cond
namespace Frame {
struct Inertial;
}
/// \endcond

namespace elliptic {
namespace Actions {

/*!
 * \brief Places the analytic solution of the system fields in the DataBox.
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 * - System:
 *   - `fields_tag`
 * - DataBox:
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `db::add_tag_prefix<::Tags::Analytic, fields_tag>`
 */
struct InitializeAnalyticSolution {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using analytic_solutions_tag =
        db::add_tag_prefix<::Tags::Analytic, fields_tag>;

    using simple_tags = db::AddSimpleTags<analytic_solutions_tag>;
    using compute_tags = db::AddComputeTags<>;

    // Compute the analytic solution for the system fields
    const auto& inertial_coords =
        get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    auto analytic_fields =
        make_with_value<db::item_type<fields_tag>>(inertial_coords, 0.);
    // This actually sets the complete set of tags in the Variables, but there
    // is no Variables constructor from a TaggedTuple (yet)
    analytic_fields.assign_subset(
        get<typename Metavariables::analytic_solution_tag>(cache).variables(
            inertial_coords, db::get_variables_tags_list<fields_tag>{}));

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeAnalyticSolution,
                                             simple_tags, compute_tags>(
            std::move(box),
            db::item_type<analytic_solutions_tag>(analytic_fields)));
  }
};

}  // namespace Actions
}  // namespace elliptic
