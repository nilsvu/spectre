// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/InterfaceHelpers.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"

namespace LinearSolver {
namespace schwarz_detail {

template <typename OptionsGroup, typename SubdomainOperator>
struct SubdomainBoundaryDataInboxTag
    : public Parallel::InboxInserters::Map<
          SubdomainBoundaryDataInboxTag<OptionsGroup, SubdomainOperator>> {
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
  //   using const_global_cache_tags = tmpl::list<Tags::Overlap<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementId<Dim>& /*element_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Parallel::printf("%s Send subdomain data in step %zu\n", element_index,
    //               get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    // using inv_jacobian_tag =
    //     ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
    //                             ::Tags::Coordinates<Dim, Frame::Logical>>;
    const auto collected_overlap_data =
        interface_apply<domain::Tags::InternalDirections<Dim>,
                        typename SubdomainOperator::collect_overlap_data>(box);

    const auto& element = get<domain::Tags::Element<Dim>>(box);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const auto& orientation = direction_and_neighbors.second.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      // Parallel::printf("Sending residual on overlap: %s\n",
      //                  residual_on_overlap);
      // Iterate over neighbors
      for (const auto& neighbor : direction_and_neighbors.second) {
        // Make a copy of the overlap data to orient and send to the neighbor
        auto overlap_data = collected_overlap_data.at(direction);
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

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct ReceiveSubdomainData {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t volume_dim = SubdomainOperator::volume_dim;
  using SubdomainDataType = typename SubdomainOperator::SubdomainDataType;
  using boundary_data_inbox_tag =
      SubdomainBoundaryDataInboxTag<OptionsGroup, SubdomainOperator>;

 public:
  using inbox_tags = tmpl::list<boundary_data_inbox_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementId<volume_dim>& /*element_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Parallel::printf("%s Receive subdomain data in step %zu\n",
    // element_index,
    //              get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    auto& inbox = tuples::get<boundary_data_inbox_tag>(inboxes);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    typename SubdomainDataType::BoundaryDataType subdomain_boundary_data{};
    if (temporal_received != inbox.end()) {
      db::mutate<Tags::SubdomainBoundaryData<SubdomainOperator, OptionsGroup>>(
          make_not_null(&box),
          [&temporal_received](
              const gsl::not_null<typename SubdomainDataType::BoundaryDataType*>
                  subdomain_boundary_data) {
            *subdomain_boundary_data = std::move(temporal_received->second);
          });
      inbox.erase(temporal_received);
    }
    return {std::move(box)};
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim>
  static bool is_ready(
      const db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_index*/) noexcept {
    return dg::has_received_from_all_mortars<boundary_data_inbox_tag>(
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
        get<domain::Tags::Element<Dim>>(box), inboxes);
  }
};

template <typename OptionsGroup, typename SubdomainOperator>
struct SubdomainBoundarySolutionsInboxTag
    : public Parallel::InboxInserters::Map<
          SubdomainBoundarySolutionsInboxTag<OptionsGroup, SubdomainOperator>> {
  static constexpr size_t volume_dim = SubdomainOperator::volume_dim;
  using temporal_id = size_t;
  using type = std::unordered_map<
      temporal_id,
      typename SubdomainOperator::SubdomainDataType::BoundaryDataType>;
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct SendOverlapSolution {
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
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementId<volume_dim>& /*element_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::SubdomainBoundaryData<SubdomainOperator, OptionsGroup>>(
        make_not_null(&box),
        [&cache](
            const gsl::not_null<typename SubdomainDataType::BoundaryDataType*>
                subdomain_boundary_data,
            const Element<volume_dim>& element, const size_t temporal_id) {
          auto& receiver_proxy =
              Parallel::get_parallel_component<ParallelComponent>(cache);
          for (auto& mortar_id_and_overlap_solution :
               *subdomain_boundary_data) {
            const auto& mortar_id = mortar_id_and_overlap_solution.first;
            const auto& direction = mortar_id.first;
            const auto& neighbor_id = mortar_id.second;
            const auto& orientation =
                element.neighbors().at(direction).orientation();
            const auto direction_from_neighbor =
                orientation(direction.opposite());
            auto& overlap_solution = mortar_id_and_overlap_solution.second;

            if (not orientation.is_aligned()) {
              overlap_solution.orient(orientation);
            }
            Parallel::receive_data<boundary_solutions_inbox_tag>(
                receiver_proxy[neighbor_id], temporal_id,
                std::make_pair(
                    std::make_pair(direction_from_neighbor, element.id()),
                    std::move(overlap_solution)));
          }
        },
        get<domain::Tags::Element<volume_dim>>(box),
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));

    return {std::move(box)};
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
      const ElementId<Dim>& /*element_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Parallel::printf("%s Receive overlap solution in step %zu\n",
    // element_index,
    //               get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    auto& inbox = tuples::get<boundary_solutions_inbox_tag>(inboxes);
    const auto& temporal_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    // const auto& logical_coords =
    //     db::get<::Tags::Coordinates<Dim, Frame::Logical>>(box);
    if (temporal_received != inbox.end()) {
      db::mutate<fields_tag>(
          make_not_null(&box),
          [&temporal_received /*, &element_index, &logical_coords*/](
              const gsl::not_null<db::item_type<fields_tag>*> fields) noexcept {
            for (const auto& mortar_id_and_overlap_solution :
                 temporal_received->second) {
              // const auto& mortar_id = mortar_id_and_overlap_solution.first;
              // const auto& direction = mortar_id.first;
              // const size_t dimension = direction.dimension();
              const auto& overlap_solution =
                  mortar_id_and_overlap_solution.second;
              // const double overlap_width = overlap_solution.overlap_width();
              // Parallel::printf("%s  Incoming overlap data from %s:\n%s\n",
              //                  element_index, mortar_id,
              //                  overlap_solution.field_data);
              // auto extended_overlap_solution =
              //     overlap_solution.extended_field_data();
              // Parallel::printf(
              //     "%s  Weighting overlap data with width %f (coming from
              //     %s)\n", element_index, overlap_width, mortar_id);
              // DataVector extended_logical_coords =
              //     logical_coords.get(dimension) - direction.sign() * 2.;
              // Parallel::printf(
              //     "%s  Extended logical coords for overlap coming from %s: "
              //     "%s\n",
              //     element_index, mortar_id, extended_logical_coords);
              // const auto w = weight(extended_logical_coords, overlap_width,
              //                       opposite(direction.side()));
              // Parallel::printf("%s  Weights:\n%s\n", element_index, w);
              // extended_overlap_solution *= w;
              // Parallel::printf(
              //     "%s  Weighted (extended) overlap data from %s:\n%s\n",
              //     element_index, mortar_id, extended_overlap_solution);
              // *fields += extended_overlap_solution;
              overlap_solution.add_to(fields);
            }
          });
      inbox.erase(temporal_received);
    }

    // Parallel::printf("%s  Updated fields: %s\n", element_index,
    //                  get<fields_tag>(box));

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim>
  static bool is_ready(
      const db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_index*/) noexcept {
    return dg::has_received_from_all_mortars<boundary_solutions_inbox_tag>(
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
        get<domain::Tags::Element<Dim>>(box), inboxes);
  }
};

}  // namespace schwarz_detail
}  // namespace LinearSolver
