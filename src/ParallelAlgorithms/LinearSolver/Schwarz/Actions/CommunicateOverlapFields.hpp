// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t Dim>
struct ElementId;
/// \endcond

namespace LinearSolver::Schwarz::Actions {

namespace detail {
template <size_t Dim, typename OverlapFields, typename OptionsGroup>
struct OverlapFieldsTag
    : public Parallel::InboxInserters::Map<
          OverlapFieldsTag<Dim, OverlapFields, OptionsGroup>> {
  using temporal_id = size_t;
  using type = std::unordered_map<
      temporal_id,
      OverlapMap<Dim, tuples::tagged_tuple_from_typelist<OverlapFields>>>;
};
}  // namespace detail

template <typename OverlapFields, typename OptionsGroup>
struct SendOverlapFields;

template <typename... OverlapFields, typename OptionsGroup>
struct SendOverlapFields<tmpl::list<OverlapFields...>, OptionsGroup> {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& element = get<domain::Tags::Element<Dim>>(box);

    // Skip communicating if the overlap is empty
    if (UNLIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) == 0 or
                 element.neighbors().size() == 0)) {
      return {std::move(box)};
    }

    // const auto& full_extents = get<domain::Tags::Mesh<Dim>>(box).extents();
    // const auto& all_intruding_extents =
    //     get<Tags::IntrudingExtents<Dim, OptionsGroup>>(box);
    const auto& iteration_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);

    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                 static_cast<int>(::Verbosity::Debug))) {
      Parallel::printf("%s " + Options::name<OptionsGroup>() +
                           "(%zu): Send overlap fields\n",
                       element_id, iteration_id);
    }

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const auto& neighbors = direction_and_neighbors.second;
      const auto direction_from_neighbor =
          neighbors.orientation()(direction.opposite());
      // const size_t intruding_extent =
      //     gsl::at(all_intruding_extents, direction.dimension());
      tuples::TaggedTuple<OverlapFields...> overlap_fields{};
      // tmpl::for_each<tmpl::list<OverlapFields...>>(
      //     [&overlap_fields, &box, &full_extents, &intruding_extent,
      //      &direction](const auto tag_v) noexcept {
      //       using tag = tmpl::type_from<decltype(tag_v)>;
      //       get<tag>(overlap_fields) =
      //       LinearSolver::Schwarz::data_on_overlap(
      //           db::get<tag>(box), full_extents, intruding_extent,
      //           direction);
      //     });
      // Don't restrict to overlap for now, since the subdomain operator works
      // with extended data
      tmpl::for_each<tmpl::list<OverlapFields...>>(
          [&overlap_fields, &box](const auto tag_v) noexcept {
            using tag = tmpl::type_from<decltype(tag_v)>;
            get<tag>(overlap_fields) = db::get<tag>(box);
          });
      size_t i = 0;
      for (const auto& neighbor : neighbors) {
        Parallel::receive_data<detail::OverlapFieldsTag<
            Dim, tmpl::list<OverlapFields...>, OptionsGroup>>(
            receiver_proxy[neighbor], iteration_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                (i + 1) < neighbors.size() ? overlap_fields
                                           : std::move(overlap_fields)));
        ++i;
      }
    }
    return {std::move(box)};
  }
};

template <size_t Dim, typename OverlapFields, typename OptionsGroup>
struct ReceiveOverlapFields;

template <size_t Dim, typename... OverlapFields, typename OptionsGroup>
struct ReceiveOverlapFields<Dim, tmpl::list<OverlapFields...>, OptionsGroup> {
 private:
  using overlap_fields_tag =
      detail::OverlapFieldsTag<Dim, tmpl::list<OverlapFields...>, OptionsGroup>;

 public:
  using inbox_tags = tmpl::list<overlap_fields_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables>
  static bool is_ready(const db::DataBox<DbTagsList>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ElementId<Dim>& /*element_id*/) noexcept {
    if (UNLIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) == 0)) {
      return true;
    }
    return dg::has_received_from_all_mortars<overlap_fields_tag>(
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
        get<domain::Tags::Element<Dim>>(box), inboxes);
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& iteration_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    const auto& element = get<domain::Tags::Element<Dim>>(box);
    if (LIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) > 0 and
               element.neighbors().size() > 0)) {
      if (UNLIKELY(static_cast<int>(
                       get<LinearSolver::Tags::Verbosity<OptionsGroup>>(box)) >=
                   static_cast<int>(::Verbosity::Debug))) {
        Parallel::printf("%s " + Options::name<OptionsGroup>() +
                             "(%zu): Receive overlap fields\n",
                         element_id, iteration_id);
      }

      auto all_overlap_fields =
          std::move(tuples::get<overlap_fields_tag>(inboxes)
                        .extract(iteration_id)
                        .mapped());
      db::mutate<Tags::Overlaps<OverlapFields, Dim, OptionsGroup>...>(
          make_not_null(&box),
          [&all_overlap_fields](const auto... local_overlap_fields) noexcept {
            for (auto& overlap_id_and_fields : all_overlap_fields) {
              const auto& overlap_id = overlap_id_and_fields.first;
              auto& overlap_fields = overlap_id_and_fields.second;
              const auto helper = [&overlap_id](const auto local_overlap_field,
                                                auto& overlap_field) noexcept {
                local_overlap_field->at(overlap_id) = std::move(overlap_field);
              };
              EXPAND_PACK_LEFT_TO_RIGHT(helper(
                  local_overlap_fields, get<OverlapFields>(overlap_fields)));
            }
          });
    }

    return {std::move(box)};
  }
};

}  // namespace LinearSolver::Schwarz::Actions
