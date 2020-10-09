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
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::Actions {

struct InitializeBackgroundFields {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using background_fields = typename system::background_fields;
    using background_fields_tag = ::Tags::Variables<background_fields>;

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    typename background_fields_tag::type background_fields_vars{num_grid_points,
                                                                0.};
    background_fields_vars.assign_subset(
        Parallel::get<typename Metavariables::boundary_conditions_tag>(cache)
            .variables(inertial_coords, background_fields{}));

    return std::make_tuple(::Initialization::merge_into_databox<
                           InitializeBackgroundFields,
                           db::AddSimpleTags<background_fields_tag>>(
        std::move(box), std::move(background_fields_vars)));
  }
};

}  // namespace elliptic::Actions
