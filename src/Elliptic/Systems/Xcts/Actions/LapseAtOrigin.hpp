// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"

#include "Parallel/Printf.hpp"

namespace Xcts {

namespace detail {

template <size_t Dim>
boost::optional<tnsr::I<DataVector, Dim, Frame::Logical>>
origin_logical_coordinates(const ElementId<Dim>& element_id,
                           const Domain<Dim, Frame::Inertial>& domain) noexcept;
template <size_t Dim>
std::array<boost::optional<tnsr::I<DataVector, Dim, Frame::Logical>>, 2>
star_centers_logical_coordinates(
    const ElementId<Dim>& element_id,
    const Domain<Dim, Frame::Inertial>& domain) noexcept;

template <size_t Dim>
std::tuple<size_t, double> lapse_at_origin(
    const DataVector& lapse,
    const boost::optional<tnsr::I<DataVector, Dim, Frame::Logical>>&
        origin_logical_coords,
    const Mesh<Dim>& mesh) noexcept;

namespace Tags {
template <size_t Dim>
struct OriginLogicalCoordinates : db::SimpleTag {
  using type = boost::optional<tnsr::I<DataVector, Dim, Frame::Logical>>;
  static std::string name() noexcept { return "OriginLogicalCoordinates"; }
};

template <size_t Dim>
struct StarCentersLogicalCoordinates : db::SimpleTag {
  using type =
      std::array<boost::optional<tnsr::I<DataVector, Dim, Frame::Logical>>, 2>;
  static std::string name() noexcept { return "StarCentersLogicalCoordinates"; }
};
}  // namespace Tags
}  // namespace detail

namespace Actions {

/// \cond
struct ReceiveLapseAtOrigin;
struct ReceiveLapseAtStarCenters;
/// \endcond

struct InitializeLapseAtOrigin {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags =
        db::AddSimpleTags<Xcts::Tags::LapseAtOrigin,
                          detail::Tags::OriginLogicalCoordinates<Dim>>;
    using compute_tags = db::AddComputeTags<>;

    auto origin_logical_coords = detail::origin_logical_coordinates(
        ElementId<Dim>{array_index},
        get<::Tags::Domain<Dim, Frame::Inertial>>(box));

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeLapseAtOrigin,
                                             simple_tags, compute_tags>(
            std::move(box), std::numeric_limits<double>::signaling_NaN(),
            std::move(origin_logical_coords)));
  }
};

struct InitializeLapseAtStarCenters {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags =
        db::AddSimpleTags<Xcts::Tags::LapseAtStarCenters,
                          detail::Tags::StarCentersLogicalCoordinates<Dim>>;
    using compute_tags = db::AddComputeTags<>;

    auto star_centers_logical_coords = detail::star_centers_logical_coordinates(
        ElementId<Dim>{array_index},
        get<::Tags::Domain<Dim, Frame::Inertial>>(box));

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeLapseAtStarCenters,
                                             simple_tags, compute_tags>(
            std::move(box),
            std::array<double, 2>{
                {std::numeric_limits<double>::signaling_NaN(),
                 std::numeric_limits<double>::signaling_NaN()}},
            std::move(star_centers_logical_coords)));
  }
};

struct UpdateLapseAtOrigin {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto lapse =
        get(get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(box)) /
        get(get<Xcts::Tags::ConformalFactor<DataVector>>(box));

    const auto multiplicity_and_lapse_at_origin = detail::lapse_at_origin(
        std::move(lapse), get<detail::Tags::OriginLogicalCoordinates<Dim>>(box),
        get<::Tags::Mesh<Dim>>(box));

    Parallel::contribute_to_reduction<ReceiveLapseAtOrigin>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::Plus<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Divides<>,
                                     std::index_sequence<0>>>{
            get<0>(multiplicity_and_lapse_at_origin),
            get<1>(multiplicity_and_lapse_at_origin)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ParallelComponent>(cache));

    return {std::move(box), true};
  }
};

struct UpdateLapseAtStarCenters {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto lapse =
        get(get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(box)) /
        get(get<Xcts::Tags::ConformalFactor<DataVector>>(box));

    const auto left_multiplicity_and_lapse = detail::lapse_at_origin(
        lapse, get<detail::Tags::StarCentersLogicalCoordinates<Dim>>(box)[0],
        get<::Tags::Mesh<Dim>>(box));
    const auto right_multiplicity_and_lapse = detail::lapse_at_origin(
        lapse, get<detail::Tags::StarCentersLogicalCoordinates<Dim>>(box)[1],
        get<::Tags::Mesh<Dim>>(box));

    Parallel::contribute_to_reduction<ReceiveLapseAtStarCenters>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::Plus<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Divides<>,
                                     std::index_sequence<0>>,
            Parallel::ReductionDatum<size_t, funcl::Plus<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Divides<>,
                                     std::index_sequence<0>>>{
            get<0>(left_multiplicity_and_lapse),
            get<1>(left_multiplicity_and_lapse),
            get<0>(right_multiplicity_and_lapse),
            get<1>(right_multiplicity_and_lapse)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ParallelComponent>(cache));

    return {std::move(box), true};
  }
};

struct ReceiveLapseAtOrigin {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Xcts::Tags::LapseAtOrigin,
                                              DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const size_t left_num_elements, const double left_lapse,
                    const size_t right_num_elements,
                    const double right_lapse) noexcept {
    Parallel::printf(
        "Lapse left: %e, num_elem: %d, lapse right: %e, num_elem: %d\n",
        left_lapse, left_num_elements, right_lapse, right_num_elements);
    db::mutate<Xcts::Tags::LapseAtStarCenters>(make_not_null(&box), [
      left_lapse, right_lapse
    ](const gsl::not_null<std::array<double, 2>*> local_lapse) noexcept {
      *local_lapse = std::array<double, 2>{{left_lapse, right_lapse}};
    });

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

struct ReceiveLapseAtStarCenters {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Xcts::Tags::LapseAtStarCenters,
                                              DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const size_t left_num_elements, const double left_lapse,
                    const size_t right_num_elements,
                    const double right_lapse) noexcept {
    Parallel::printf(
        "Lapse left: %e, num_elem: %d, lapse right: %e, num_elem: %d\n",
        left_lapse, left_num_elements, right_lapse, right_num_elements);
    db::mutate<Xcts::Tags::LapseAtStarCenters>(make_not_null(&box), [
      left_lapse, right_lapse
    ](const gsl::not_null<std::array<double, 2>*> local_lapse) noexcept {
      *local_lapse = std::array<double, 2>{{left_lapse, right_lapse}};
    });

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

}  // namespace Actions
}  // namespace Xcts
