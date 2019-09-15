// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Tags {
template <typename Metavariables>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace Xcts {

namespace Tags {

struct StarCenters : db::SimpleTag {
  using type = std::array<double, 2>;
  static std::string name() noexcept { return "StarCenters"; }
};

template <typename OptionTag>
struct StarCentersFromOption : StarCenters {
  using type = std::array<double, 2>;
  using option_tags = tmpl::list<OptionTag>;
  static type create_from_options(const typename OptionTag::type& option) {
    return option.star_centers();
  }
};
}  // namespace Tags

// namespace OptionHolders {
// /// A line segment extending from `Begin` to `End`,
// /// containing `NumberOfPoints` uniformly-spaced points including the
// endpoints.
// ///
// /// \note Input coordinates are interpreted in the frame given by
// /// `Metavariables::domain_frame`
// template <size_t VolumeDim>
// struct LineSegment {
//   struct Begin {
//     using type = std::array<double, VolumeDim>;
//     static constexpr OptionString help = {"Beginning endpoint"};
//   };
//   struct End {
//     using type = std::array<double, VolumeDim>;
//     static constexpr OptionString help = {"Ending endpoint"};
//   };
//   struct NumberOfPoints {
//     using type = size_t;
//     static constexpr OptionString help = {
//         "Number of points including endpoints"};
//     static type lower_bound() noexcept { return 2; }
//   };
//   using options = tmpl::list<Begin, End, NumberOfPoints>;
//   static constexpr OptionString help = {
//       "A line segment extending from Begin to End, containing NumberOfPoints"
//       " uniformly-spaced points including the endpoints."};

//   LineSegment(std::array<double, VolumeDim> begin_in,
//               std::array<double, VolumeDim> end_in,
//               size_t number_of_points_in) noexcept;

//   LineSegment() = default;
//   LineSegment(const LineSegment& /*rhs*/) = delete;
//   LineSegment& operator=(const LineSegment& /*rhs*/) = delete;
//   LineSegment(LineSegment&& /*rhs*/) noexcept = default;
//   LineSegment& operator=(LineSegment&& /*rhs*/) noexcept = default;
//   ~LineSegment() = default;

//   // clang-tidy non-const reference pointer.
//   void pup(PUP::er& p) noexcept;  // NOLINT

//   std::array<double, VolumeDim> begin{};
//   std::array<double, VolumeDim> end{};
//   size_t number_of_points{};
// };

// template <size_t VolumeDim>
// bool operator==(const LineSegment<VolumeDim>& lhs,
//                 const LineSegment<VolumeDim>& rhs) noexcept;
// template <size_t VolumeDim>
// bool operator!=(const LineSegment<VolumeDim>& lhs,
//                 const LineSegment<VolumeDim>& rhs) noexcept;

// }  // namespace OptionHolders

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sends points on a line segment to an `Interpolator`.
///
/// Uses:
/// - DataBox:
///   - `::Tags::Domain<VolumeDim, Frame>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag, typename StarCentersTag,
          size_t VolumeDim>
struct SendStarCentersToInterpolator {
  using const_global_cache_tags = tmpl::list<>;
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, ::intrp::Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    const auto& star_centers =
        Parallel::get<StarCentersTag>(cache).star_centers();

    tnsr::I<DataVector, VolumeDim, typename Metavariables::domain_frame>
        target_points{size_t{2}, 0.};
    get<0>(target_points) = DataVector{star_centers[0], star_centers[1]};

    ::intrp::send_points_to_interpolator<InterpolationTargetTag>(
        box, cache, std::move(target_points), temporal_id);
  }
};

}  // namespace Actions
}  // namespace Xcts
