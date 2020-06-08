// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Informer/LogActions.hpp"
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/InterMeshOperators.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/MeshHierarchy.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t Dim>
struct ElementId;
/// \endcond

namespace LinearSolver::multigrid {

template <size_t Dim, typename ReceiveTag>
struct DataFromChildrenInboxTag
    : public Parallel::InboxInserters::Map<
          DataFromChildrenInboxTag<Dim, ReceiveTag>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, FixedHashMap<two_to_the(Dim), ElementId<Dim>,
                                         typename ReceiveTag::type,
                                         boost::hash<ElementId<Dim>>>>;
};

namespace Actions {

template <typename FieldsTag, typename OptionsGroup,
          typename ReceiveTag = FieldsTag>
struct SendFieldsToCoarserGrid {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Skip restriction on coarsest level
    const auto& parent_id = get<Tags::ParentElementId<Dim>>(box);
    if (not parent_id) {
      return {std::move(box)};
    }

    const auto& temporal_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s " + Options::name<OptionsGroup>() +
                           "(%zu): Send fields to coarser grid\n",
                       element_id, temporal_id);
    }

    // TODO: Move jacobian ratio into restriction operator
    // TODO: Make sure the jacobians are handled correctly on curved meshes
    // TODO: Resolve the jacobian by numerically integrating the mass matrix
    // including the jacobian.
    auto fields = typename ReceiveTag::type(db::get<FieldsTag>(box));
    if constexpr (not Metavariables::massive_operator) {
      fields *= get(
          db::get<domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>>(
              box));
    }

    // Restrict the residual to the coarser (parent) grid and treat as source.
    // We restrict before sending the data so the restriction operation is
    // parellelized. The parent only needs to sum up all child contributions.
    // TODO: Do nothing when parent is the same element
    auto restricted_fields = apply_matrices(
        db::get<Tags::RestrictionOperator<Dim, OptionsGroup>>(box), fields,
        // TODO: make sure apply_matrices works for non-square matrices
        db::get<domain::Tags::Mesh<Dim>>(box).extents());

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    Parallel::receive_data<DataFromChildrenInboxTag<Dim, ReceiveTag>>(
        receiver_proxy[*parent_id], temporal_id,
        std::make_pair(element_id, std::move(restricted_fields)),
        // We re-start the algorithm on coarser levels when they
        // `receive_data`, since it is terminated in `PrepareSolve`.
        true);
    return {std::move(box)};
  }
};

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename ReceiveTag = FieldsTag>
struct ReceiveFieldsFromFinerGrid {
  using inbox_tags = tmpl::list<DataFromChildrenInboxTag<Dim, ReceiveTag>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ElementId<Dim>& /*element_id*/) noexcept {
    const auto& child_ids = get<Tags::ChildElementIds<Dim>>(box);
    if (child_ids.empty()) {
      return true;
    }
    const auto& inbox =
        tuples::get<DataFromChildrenInboxTag<Dim, ReceiveTag>>(inboxes);
    const auto& temporal_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    if (temporal_received == inbox.end()) {
      return false;
    }
    const auto& received_children_data = temporal_received->second;
    for (const auto& child_id : child_ids) {
      if (received_children_data.find(child_id) ==
          received_children_data.end()) {
        return false;
      }
    }
    return true;
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Skip on finest grid
    const auto& child_ids = get<Tags::ChildElementIds<Dim>>(box);
    if (child_ids.empty()) {
      return {std::move(box)};
    }

    const auto& temporal_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s " + Options::name<OptionsGroup>() +
                           "(%zu): Receive fields from finer grid\n",
                       element_id, temporal_id);
    }

    auto children_data = std::move(
        tuples::get<DataFromChildrenInboxTag<Dim, ReceiveTag>>(inboxes)
            .extract(temporal_id)
            .mapped());

    // Assemble restricted data from children
    // TODO: Specialize for when single child is the same element
    db::mutate<ReceiveTag>(
        make_not_null(&box),
        [&children_data](const auto source, const Mesh<Dim>& mesh,
                         const Scalar<DataVector>& det_jacobian) noexcept {
          *source = typename ReceiveTag::type{mesh.number_of_grid_points(), 0.};
          for (auto& child_id_and_data : children_data) {
            *source += child_id_and_data.second;
          }
          if constexpr (not Metavariables::massive_operator) {
            *source /= get(det_jacobian);
          }
        },
        db::get<domain::Tags::Mesh<Dim>>(box),
        db::get<domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>>(
            box));

    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace LinearSolver::multigrid
