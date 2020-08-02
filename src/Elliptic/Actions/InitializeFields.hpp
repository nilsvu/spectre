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
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

/*!
 * \brief Initializes the DataBox tags related to the elliptic system
 *
 * The system fields are initially set to zero.
 *
 * \note Currently the sources are always retrieved from an analytic solution.
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 * - System:
 *   - `fields_tag`
 *   - `primal_fields`
 * - DataBox:
 *   - `Tags::Mesh<Dim>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *   - `db::add_tag_prefix<::Tags::FixedSource, fields_tag>`
 */
struct InitializeFields {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;

    typename fields_tag::type fields{variables_from_tagged_tuple(
        db::get<typename Metavariables::initial_guess_tag>(box).variables(
            db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box),
            typename db::add_tag_prefix<::Tags::Initial,
                                        fields_tag>::tags_list{}))};

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeFields,
                                             db::AddSimpleTags<fields_tag>>(
            std::move(box), std::move(fields)));
  }
};

}  // namespace Actions
}  // namespace elliptic
