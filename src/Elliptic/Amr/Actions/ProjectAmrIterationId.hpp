// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::amr::Actions {

/*!
 * \brief AMR projector for tags describing the elliptic AMR state.
 *
 * Copies the AMR iteration ID from the parent or children during h-refinement.
 */
struct ProjectAmrIterationId : tt::ConformsTo<::amr::protocols::Projector> {
  using return_tags =
      tmpl::list<Convergence::Tags::IterationId<::amr::OptionTags::AmrGroup>,
                 Convergence::Tags::HasConverged<::amr::OptionTags::AmrGroup>>;
  using argument_tags = tmpl::list<>;

  // p-refinement
  template <size_t Dim>
  static void apply(
      gsl::not_null<size_t*> /*amr_iteration_id*/,
      const gsl::not_null<Convergence::HasConverged*> /*amr_has_converged*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {}

  // h-refinement
  template <typename... ParentTags>
  static void apply(
      const gsl::not_null<size_t*> amr_iteration_id,
      const gsl::not_null<Convergence::HasConverged*> /*amr_has_converged*/,
      const tuples::TaggedTuple<ParentTags...>& parent_items) {
    *amr_iteration_id =
        get<Convergence::Tags::IterationId<::amr::OptionTags::AmrGroup>>(
            parent_items);
  }

  // h-coarsening
  template <size_t Dim, typename... ChildTags>
  static void apply(
      const gsl::not_null<size_t*> amr_iteration_id,
      const gsl::not_null<Convergence::HasConverged*> /*amr_has_converged*/,
      const std::unordered_map<
          ElementId<Dim>, tuples::TaggedTuple<ChildTags...>>& children_items) {
    *amr_iteration_id =
        get<Convergence::Tags::IterationId<::amr::OptionTags::AmrGroup>>(
            children_items.begin()->second);
  }
};

}  // namespace elliptic::amr::Actions
