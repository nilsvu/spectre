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
  using type = std::unordered_map<
      temporal_id,
      typename SubdomainOperator::SubdomainDataType::BoundaryDataType>;
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct SendSubdomainData {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using boundary_data_inbox_tag =
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
    const auto& overlap = get<Tags::Overlap<OptionsGroup>>(box);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& orientation = direction_and_neighbors.second.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      // Construct the data on the overlap with the neighbor
      const auto overlap_extents =
          LinearSolver::schwarz_detail::overlap_extents(mesh.extents(), overlap,
                                                        dimension);
      const auto residual_on_overlap = data_on_overlap(
          get<residual_tag>(box), mesh.extents(), overlap_extents, direction);
      Parallel::printf("Sending residual on overlap: %s\n",
                       residual_on_overlap);
      // Iterate over neighbors
      for (const auto& neighbor : direction_and_neighbors.second) {
        // Construct the data to send
        auto overlap_data =
            typename SubdomainOperator::SubdomainDataType::OverlapDataType{
                typename SubdomainOperator::SubdomainDataType::Vars(
                    residual_on_overlap),
                mesh,
                get<inv_jacobian_tag>(box),
                direction,
                magnitude_of_face_normals.at(direction),
                overlap_extents};
        if (not orientation.is_aligned()) {
          overlap_data.orient(orientation);
        }
        Parallel::receive_data<boundary_data_inbox_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                std::move(overlap_data)));
      }
    }
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename InboxTag, size_t Dim, typename TemporalIdType,
          typename... InboxTags>
bool has_received_from_all_neighbors(
    const TemporalIdType& temporal_id, const Element<Dim>& element,
    const tuples::TaggedTuple<InboxTags...>& inboxes) noexcept {
  if (element.number_of_neighbors() == 0) {
    return true;
  }
  const auto& inbox = tuples::get<InboxTag>(inboxes);
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

template <typename OptionsGroup, typename SubdomainOperator>
struct SubdomainBoundarySolutionsInboxTag {
  static constexpr size_t volume_dim = SubdomainOperator::volume_dim;
  using temporal_id = size_t;
  using type = std::unordered_map<
      temporal_id,
      typename SubdomainOperator::SubdomainDataType::BoundaryDataType>;
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
  using boundary_data_inbox_tag =
      SubdomainBoundaryDataInboxTag<OptionsGroup, SubdomainOperator>;
  using boundary_solutions_inbox_tag =
      SubdomainBoundarySolutionsInboxTag<OptionsGroup, SubdomainOperator>;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Iterations<OptionsGroup>>;
  using inbox_tags = tmpl::list<boundary_data_inbox_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<volume_dim>& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Receive subdomain data in step %zu\n", element_index,
                     get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    auto& inbox = tuples::get<boundary_data_inbox_tag>(inboxes);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    typename SubdomainDataType::BoundaryDataType subdomain_boundary_data{};
    if (temporal_received != inbox.end()) {
      subdomain_boundary_data = std::move(temporal_received->second);
      inbox.erase(temporal_received);
    }

    Parallel::printf("%s Perform Schwarz step %zu\n", element_index,
                     get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));

    // Gather residual and overlap from neighbors
    const SubdomainDataType residual_subdomain{
        typename SubdomainDataType::Vars(get<residual_tag>(box)),
        std::move(subdomain_boundary_data)};

    Parallel::printf("%s  Initial fields: %s\n", element_index,
                     get<fields_tag>(box));
    Parallel::printf("%s  Residual (central): %s\n", element_index,
                     residual_subdomain.element_data);
    Parallel::printf("%s  Overlap with: %d elements\n", element_index,
                     residual_subdomain.boundary_data.size());

    const auto& subdomain_solver =
        get<Tags::SubdomainSolverBase<OptionsGroup>>(box);
    auto subdomain_solution = subdomain_solver(
        [&box](const SubdomainDataType& arg) noexcept {
          return db::apply<SubdomainOperator>(box, arg);
        },
        residual_subdomain,
        // Using the residual as initial guess so not iterating the
        // subdomain solver at all is the identity operation
        residual_subdomain);

    Parallel::printf("%s  Subdomain solution (central): %s\n", element_index,
                     subdomain_solution.element_data);

    // Weighting
    // The central element will receive overlap contributions from its face
    // neighbors, so we weight the subdomain solution with each neighbor's
    // _incoming_ overlap width.
    // TODO: Is this the correct way to handle h-refined mortars?
    // TODO: The overlap width we'll receive may be different to the overlap
    // we're sending because of p-refinement. Should we weight with the expected
    // incoming contribution's width or with the one we're sending?
    // TODO: We'll have to keep in mind that the weighting operation should
    // preserve symmetry of the linear operator
    const auto& element = db::get<::Tags::Element<volume_dim>>(box);
    const auto& logical_coords =
        db::get<::Tags::Coordinates<volume_dim, Frame::Logical>>(box);
    // const auto& mesh = db::get<::Tags::Mesh<volume_dim>>(box);
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_and_neighbors.second;
      const auto& logical_coord = logical_coords.get(dimension);
      for (const auto& neighbor_id : neighbors_in_direction) {
        const auto mortar_id = std::make_pair(direction, neighbor_id);
        const auto& overlap_solution =
            subdomain_solution.boundary_data.at(mortar_id);
        // Use incoming or outgoing overlap width here?
        // const double overlap_width_in_center = overlap_width(
        //     mesh.slice_through(dimension),
        //     overlap_extent(
        //         mesh.extents(dimension),
        //         get<LinearSolver::Tags::Overlap<OptionsGroup>>(box)));
        const double overlap_width_in_center = overlap_solution.overlap_width();
        Parallel::printf(
            "%s  Weighting center with width %f for overlap with %s\n",
            element_index, overlap_width_in_center, mortar_id);
        Parallel::printf("%s  Logical coords for overlap with %s: %s\n",
                         element_index, mortar_id, logical_coord);
        const auto w =
            weight(logical_coord, overlap_width_in_center, direction.side());
        Parallel::printf("%s  Weights:\n%s\n", element_index, w);
        subdomain_solution.element_data *= w;
      }
    }

    Parallel::printf("%s  Subdomain solution WEIGHTED (central): %s\n",
                     element_index, subdomain_solution.element_data);

    // Send overlap data to neighbors
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (auto& mortar_id_and_overlap_solution :
         subdomain_solution.boundary_data) {
      const auto& mortar_id = mortar_id_and_overlap_solution.first;
      const auto& direction = mortar_id.first;
      const auto& neighbor_id = mortar_id.second;
      const auto& orientation = element.neighbors().at(direction).orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      auto& overlap_solution = mortar_id_and_overlap_solution.second;

      if (not orientation.is_aligned()) {
        overlap_solution.orient(orientation);
      }
      Parallel::receive_data<boundary_solutions_inbox_tag>(
          receiver_proxy[neighbor_id], temporal_id,
          std::make_pair(std::make_pair(direction_from_neighbor, element.id()),
                         std::move(overlap_solution)));
    }

    // Apply solution to central element
    db::mutate<fields_tag>(
        make_not_null(&box),
        [&subdomain_solution](
            const gsl::not_null<db::item_type<fields_tag>*> fields) noexcept {
          *fields += subdomain_solution.element_data;
        });

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

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim>
  static bool is_ready(
      const db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*element_index*/) noexcept {
    return has_received_from_all_neighbors<boundary_data_inbox_tag>(
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
        get<::Tags::Element<Dim>>(box), inboxes);
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct ReceiveOverlapSolution {
 private:
  using fields_tag = FieldsTag;
  using boundary_solutions_inbox_tag =
      SubdomainBoundarySolutionsInboxTag<OptionsGroup, SubdomainOperator>;

 public:
  using inbox_tags = tmpl::list<boundary_solutions_inbox_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Receive overlap solution in step %zu\n", element_index,
                     get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    auto& inbox = tuples::get<boundary_solutions_inbox_tag>(inboxes);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    const auto& logical_coords =
        db::get<::Tags::Coordinates<Dim, Frame::Logical>>(box);
    if (temporal_received != inbox.end()) {
      db::mutate<fields_tag>(
          make_not_null(&box),
          [&temporal_received, &element_index, &logical_coords](
              const gsl::not_null<db::item_type<fields_tag>*> fields) noexcept {
            for (const auto& mortar_id_and_overlap_solution :
                 temporal_received->second) {
              const auto& mortar_id = mortar_id_and_overlap_solution.first;
              const auto& direction = mortar_id.first;
              const size_t dimension = direction.dimension();
              const auto& overlap_solution =
                  mortar_id_and_overlap_solution.second;
              const double overlap_width = overlap_solution.overlap_width();
              Parallel::printf("%s  Incoming overlap data from %s:\n%s\n",
                               element_index, mortar_id,
                               overlap_solution.field_data);
              auto extended_overlap_solution =
                  overlap_solution.extended_field_data();
              Parallel::printf(
                  "%s  Weighting overlap data with width %f (coming from %s)\n",
                  element_index, overlap_width, mortar_id);
              DataVector extended_logical_coords =
                  logical_coords.get(dimension) - direction.sign() * 2.;
              Parallel::printf(
                  "%s  Extended logical coords for overlap coming from %s: "
                  "%s\n",
                  element_index, mortar_id, extended_logical_coords);
              const auto w = weight(extended_logical_coords, overlap_width,
                                    opposite(direction.side()));
              Parallel::printf("%s  Weights:\n%s\n", element_index, w);
              extended_overlap_solution *= w;
              Parallel::printf(
                  "%s  Weighted (extended) overlap data from %s:\n%s\n",
                  element_index, mortar_id, extended_overlap_solution);
              *fields += extended_overlap_solution;
            }
          });
      inbox.erase(temporal_received);
    }

    Parallel::printf("%s  Updated fields: %s\n", element_index,
                     get<fields_tag>(box));

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim>
  static bool is_ready(
      const db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*element_index*/) noexcept {
    return has_received_from_all_neighbors<boundary_solutions_inbox_tag>(
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
        get<::Tags::Element<Dim>>(box), inboxes);
  }
};

}  // namespace schwarz_detail
}  // namespace LinearSolver
