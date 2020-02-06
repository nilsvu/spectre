// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

#include "Parallel/Printf.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver {
namespace schwarz_detail {

template <typename FieldsTag, typename OptionsGroup>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Prepare Schwarz solve\n", array_index);
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id) noexcept {
          *iteration_id = std::numeric_limits<size_t>::max();
        });
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename FieldsTag, typename OptionsGroup>
struct PrepareStep {
 private:
  using fields_tag = FieldsTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf(
        "%s Prepare Schwarz step %zu\n", array_index,
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box) + 1);
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>,
               LinearSolver::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t& max_iterations) noexcept {
          (*iteration_id)++;
          *has_converged = Convergence::HasConverged{
              {max_iterations, 0., 0.}, *iteration_id, 1., 1.};
        },
        get<LinearSolver::Tags::Iterations<OptionsGroup>>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename OptionsGroup, typename SubdomainOperator>
struct SubdomainBoundaryDataInboxTag {
  static constexpr size_t volume_dim = SubdomainOperator::volume_dim;
  using temporal_id = size_t;
  using type =
      std::unordered_map<temporal_id, db::item_type<Tags::SubdomainBoundaryData<
                                          OptionsGroup, SubdomainOperator>>>;
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct SendSubdomainData {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using inbox_tag =
      SubdomainBoundaryDataInboxTag<OptionsGroup, SubdomainOperator>;

 public:
  using const_global_cache_tags = tmpl::list<Tags::Overlap<OptionsGroup>>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<Dim>& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Send subdomain data in step %zu\n", element_index,
                     get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    using inv_jacobian_tag =
        ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                                ::Tags::Coordinates<Dim, Frame::Logical>>;

    const auto& element = get<::Tags::Element<Dim>>(box);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto& mesh = get<::Tags::Mesh<Dim>>(box);
    const auto& magnitude_of_face_normals = get<::Tags::Interface<
        ::Tags::InternalDirections<Dim>,
        ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>>(box);
    const auto& mortar_meshes =
        get<::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>>(box);
    const auto& mortar_sizes =
        get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(box);
    const auto& overlap = get<Tags::Overlap<OptionsGroup>>(box);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& orientation = direction_and_neighbors.second.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      // Construct the data on the overlap with the neighbor
      auto overlap_extents = mesh.extents();
      overlap_extents[dimension] = overlap;
      db::item_type<residual_tag> residual_on_overlap{overlap_extents.product(),
                                                      0.};
      for (size_t i = 0; i < overlap; i++) {
        add_slice_to_data(
            make_not_null(&residual_on_overlap),
            data_on_slice(get<residual_tag>(box), mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), direction, i)),
            overlap_extents, dimension,
            index_to_slice_at(overlap_extents, direction, i));
      }
      Parallel::printf("Sending residual on overlap: %s\n",
                       residual_on_overlap);
      // Iterate over neighbors
      for (const auto& neighbor : direction_and_neighbors.second) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        // Construct the data to send
        auto overlap_data =
            typename SubdomainOperator::SubdomainDataType::OverlapDataType{
                typename SubdomainOperator::SubdomainDataType::Vars(
                    residual_on_overlap),
                mesh,
                get<inv_jacobian_tag>(box),
                magnitude_of_face_normals.at(direction),
                overlap_extents,
                mortar_meshes.at(mortar_id),
                mortar_sizes.at(mortar_id)};
        Parallel::receive_data<inbox_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                std::move(overlap_data)));
      }
    }
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct ReceiveSubdomainData {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using inbox_tag =
      SubdomainBoundaryDataInboxTag<OptionsGroup, SubdomainOperator>;
  using subdomain_boundary_data_tag =
      Tags::SubdomainBoundaryData<OptionsGroup, SubdomainOperator>;

 public:
  using inbox_tags = tmpl::list<inbox_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Receive subdomain data in step %zu\n", element_index,
                     get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    auto& inbox = tuples::get<inbox_tag>(inboxes);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    if (temporal_received != inbox.end()) {
      db::mutate<subdomain_boundary_data_tag>(
          make_not_null(&box),
          [&temporal_received](
              const gsl::not_null<db::item_type<subdomain_boundary_data_tag>*>
                  subdomain_boundary_data) noexcept {
            *subdomain_boundary_data = std::move(temporal_received->second);
          });
      inbox.erase(temporal_received);
    }
    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim>
  static bool is_ready(
      const db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*element_index*/) noexcept {
    const auto& element = get<::Tags::Element<Dim>>(box);
    if (element.number_of_neighbors() == 0) {
      return true;
    }
    const auto& inbox = tuples::get<inbox_tag>(inboxes);
    // Check that we have received data from all neighbors for this iteration
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    if (temporal_received == inbox.end()) {
      return false;
    }
    const auto& received_neighbor_data = temporal_received->second;
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      for (const auto& neighbor : direction_and_neighbors.second) {
        const auto neighbor_received =
            received_neighbor_data.find(std::make_pair(direction, neighbor));
        if (neighbor_received == received_neighbor_data.end()) {
          return false;
        }
      }
    }
    return true;
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename SourceTag>
struct PerformStep {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t volume_dim = SubdomainOperator::volume_dim;
  using SubdomainDataType = typename SubdomainOperator::SubdomainDataType;
  using subdomain_boundary_data_tag =
      Tags::SubdomainBoundaryData<OptionsGroup, SubdomainOperator>;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Iterations<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<volume_dim>& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Perform Schwarz step %zu\n", element_index,
                     get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));

    // Gather residual and overlap from neighbors
    const SubdomainDataType residual_subdomain{
        typename SubdomainDataType::Vars(get<residual_tag>(box)),
        get<subdomain_boundary_data_tag>(box)};

    db::mutate_apply<
        tmpl::list<fields_tag, Tags::SubdomainSolverBase<OptionsGroup>>,
        tmpl::append<typename SubdomainOperator::argument_tags>>(
        [&element_index, &residual_subdomain](
            const gsl::not_null<db::item_type<fields_tag>*> fields,
            const gsl::not_null<db::item_type<
                Tags::SubdomainSolverBase<OptionsGroup>, DbTagsList>*>
                subdomain_solver,
            const auto&... args) noexcept {
          Parallel::printf("%s  Initial fields: %s\n", element_index, *fields);
          Parallel::printf("%s  Residual (central): %s\n", element_index,
                           residual_subdomain.element_data);
          Parallel::printf("%s  Overlap with: %d elements\n", element_index,
                           residual_subdomain.boundary_data.size());
          const auto delta_fields_subdomain = (*subdomain_solver)(
              [&args...](const SubdomainDataType& arg) noexcept {
                return SubdomainOperator::apply(arg, args...);
              },
              residual_subdomain,
              // Using the residual as initial guess so not iterating the
              // subdomain solver at all is the identity operation
              residual_subdomain);
          // TODO: transpose-restrict from subdomain to full domain (includes
          // weighting and sending the data to neighbors)
          *fields += delta_fields_subdomain.element_data;
          Parallel::printf("%s  Updated fields: %s\n", element_index, *fields);
        },
        make_not_null(&box));

    db::mutate<LinearSolver::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t& max_iterations, const size_t& iteration_id) noexcept {
          // Run the solver for a set number of iterations
          *has_converged = Convergence::HasConverged{
              {max_iterations, 0., 0.}, iteration_id + 1, 1., 1.};
        },
        get<LinearSolver::Tags::Iterations<OptionsGroup>>(box),
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace schwarz_detail
}  // namespace LinearSolver
