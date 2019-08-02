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

struct InitializeBackgroundFields {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using background_fields_tag =
        ::Tags::Variables<typename system::background_fields>;
    using simple_tags = db::AddSimpleTags<background_fields_tag>;
    using compute_tags = db::AddComputeTags<>;

    // Compute the analytic solution for the system fields
    const auto& inertial_coords =
        get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    auto background_fields = variables_from_tagged_tuple(
        get<typename Metavariables::analytic_solution_tag>(cache).variables(
            inertial_coords,
            db::get_variables_tags_list<background_fields_tag>{}));

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeBackgroundFields,
                                             simple_tags, compute_tags>(
            std::move(box), std::move(background_fields)));
  }
};

}  // namespace Actions
}  // namespace elliptic
