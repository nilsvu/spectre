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
}  // namespace Tags
}  // namespace detail

namespace Actions {

/// \cond
struct ReceiveLapseAtOrigin;
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

struct UpdateLapseAtOrigin {
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

struct ReceiveLapseAtOrigin {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Xcts::Tags::LapseAtOrigin,
                                              DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const size_t num_elements_at_origin,
                    const double lapse_at_origin) noexcept {
    Parallel::printf("Lapse at origin: %e, num_elem: %d\n", lapse_at_origin,
                     num_elements_at_origin);
    db::mutate<Xcts::Tags::LapseAtOrigin>(
        make_not_null(&box), [lapse_at_origin](
                                 const gsl::not_null<double*>
                                     local_lapse_at_origin) noexcept {
          *local_lapse_at_origin = lapse_at_origin;
        });

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

}  // namespace Actions
}  // namespace Xcts
