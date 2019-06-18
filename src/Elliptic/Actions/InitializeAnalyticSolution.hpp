// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic {
namespace Actions {

/*!
 * \brief Place the analytic solution of the system fields in the DataBox.
 *
 * Uses:
 * - DataBox:
 *   - `AnalyticSolutionTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `db::wrap_tags_in<::Tags::Analytic, AnalyticSolutionFields>`
 */
template <typename AnalyticSolutionTag, typename AnalyticSolutionFields>
struct InitializeAnalyticSolution {
  using const_global_cache_tags = tmpl::list<AnalyticSolutionTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using analytic_fields_tag =
        db::add_tag_prefix<::Tags::Analytic,
                           ::Tags::Variables<AnalyticSolutionFields>>;

    const auto& inertial_coords =
        get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    db::item_type<analytic_fields_tag> analytic_fields{
        variables_from_tagged_tuple(get<AnalyticSolutionTag>(cache).variables(
            inertial_coords, AnalyticSolutionFields{}))};

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeAnalyticSolution, db::AddSimpleTags<analytic_fields_tag>>(
            std::move(box), std::move(analytic_fields)));
  }
};

}  // namespace Actions
}  // namespace elliptic
