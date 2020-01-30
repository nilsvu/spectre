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

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct SubdomainBoundaryDataInboxTag {
  static constexpr size_t volume_dim = SubdomainOperator::volume_dim;
  using temporal_id = size_t;
  using type =
      std::unordered_map<temporal_id,
                         db::item_type<Tags::SubdomainBoundaryData<
                             FieldsTag, OptionsGroup, SubdomainOperator>>>;
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct SendSubdomainData {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using inbox_tag =
      SubdomainBoundaryDataInboxTag<FieldsTag, OptionsGroup, SubdomainOperator>;

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

    const auto& element = get<::Tags::Element<Dim>>(box);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    // const auto& mesh = get<::Tags::Mesh<Dim>>(box);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      // const size_t dimension = direction.dimension();
      const auto& orientation = direction_and_neighbors.second.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor : direction_and_neighbors.second) {
        // Construct the data to send
        // TODO: Make this a custom data type that includes information about
        // the mesh for the DG operator
        // Sending the raw residual data for now
        auto residual_on_overlap = data_on_overlap(get<residual_tag>(box));
        // Orient data
        if (not orientation.is_aligned()) {
          residual_on_overlap = orient_data_on_overlap(residual_on_overlap);
        }
        Parallel::receive_data<inbox_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                std::move(residual_on_overlap)));
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
      SubdomainBoundaryDataInboxTag<FieldsTag, OptionsGroup, SubdomainOperator>;
  using subdomain_boundary_data_tag =
      Tags::SubdomainBoundaryData<FieldsTag, OptionsGroup, SubdomainOperator>;

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
    db::mutate<subdomain_boundary_data_tag>(
        make_not_null(&box),
        [&temporal_received](
            const gsl::not_null<db::item_type<subdomain_boundary_data_tag>*>
                subdomain_boundary_data) noexcept {
          *subdomain_boundary_data = std::move(temporal_received->second);
        });
    inbox.erase(temporal_received);
    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim>
  static bool is_ready(
      const db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*element_index*/) noexcept {
    const auto& inbox = tuples::get<inbox_tag>(inboxes);
    // Check that we have received data from all neighbors for this iteration
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    if (temporal_received == inbox.end()) {
      return false;
    }
    const auto& received_neighbor_data = temporal_received->second;
    const auto& element = get<::Tags::Element<Dim>>(box);
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
      Tags::SubdomainBoundaryData<FieldsTag, OptionsGroup, SubdomainOperator>;

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

    // Gather residual overlap from neighbors
    typename SubdomainDataType::BoundaryDataType boundary_data{};
    for (const auto& id_and_data : get<subdomain_boundary_data_tag>(box)){
      boundary_data[id_and_data.first] = id_and_data.second;
    }
    const SubdomainDataType residual_subdomain{
        typename SubdomainDataType::Vars(get<residual_tag>(box)),
        boundary_data};

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
