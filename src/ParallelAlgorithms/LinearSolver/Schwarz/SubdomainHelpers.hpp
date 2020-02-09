// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "Domain/Side.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Math.hpp"

namespace LinearSolver {
namespace schwarz_detail {

// @{
/*!
 * \brief Construct extents that represent an overlap region within the
 * `volume_extents`
 *
 * Only the overlap extents in the given `dimension` are different to the
 * `volume_extents`. In that dimension, the overlap extent is the largest number
 * under these constraints:
 *
 * - It is at most `max_overlap`.
 * - It is smaller than the `volume_extents`.
 *
 * This means the overlap extents are always smaller than the `volume_extents`.
 * The reason for this constraint is that we define the _width_ of the overlap
 * as the element-logical coordinate distance from the face of the element to
 * the first collocation point _outside_ the overlap extents. Therefore, even an
 * overlap region that covers the full element in width does not include the
 * slice of collocation points on the opposite side of the element.
 *
 * TODO: Check this. Perhaps allow full `volume_extents` and keep width the
 * same. See if we need to take more boundary contributions into account then.
 */
size_t overlap_extent(const size_t volume_extent,
                      const size_t max_overlap) noexcept {
  return std::min(max_overlap, volume_extent - 1);
}

template <size_t Dim>
Index<Dim> overlap_extents(const Index<Dim>& volume_extents,
                           const size_t max_overlap,
                           const size_t dimension) noexcept {
  auto overlap_extents = volume_extents;
  overlap_extents[dimension] =
      overlap_extent(volume_extents[dimension], max_overlap);
  return overlap_extents;
}
// @}

double overlap_width(const Mesh<1>& overlapped_mesh,
                     const size_t overlap_extent) noexcept {
  return Spectral::collocation_points(overlapped_mesh)[overlap_extent] + 1.;
}

template <typename DataType>
DataType smoothstep(const DataType& arg) noexcept {
  static const std::vector<double> coeffs{0., 15., 0., -10., 0., 3.};
  DataType result = make_with_value<DataType>(arg, 0.);
  for (size_t i = 0; i < get_size(arg); i++) {
    get_element(result, i) =
        get_element(arg, i) > 1.
            ? 1.
            : get_element(arg, i) < -1.
                  ? -1.
                  : 0.125 * evaluate_polynomial(coeffs, get_element(arg, i));
  }
  return result;
}

template <typename DataType>
DataType weight(const DataType& logical_coord, const double width) noexcept {
  return 0.5 * (smoothstep((logical_coord + 1) / width) -
                smoothstep((logical_coord - 1) / width));
}

template <typename DataType>
DataType weight(const DataType& logical_coord, const double width,
                const Side& side) noexcept {
  const double sign = side == Side::Lower ? -1. : 1.;
  return 0.5 *
         (1. - sign * smoothstep(DataType((logical_coord - sign) / width)));
}

template <size_t Dim, typename TagsList>
Variables<TagsList> data_on_overlap(const Variables<TagsList>& volume_data,
                                    const Index<Dim>& volume_extents,
                                    const Index<Dim>& overlap_extents,
                                    const Direction<Dim>& direction) noexcept {
  Variables<TagsList> overlap_data{overlap_extents.product(), 0.};
  const size_t dimension = direction.dimension();
  for (size_t i = 0; i < overlap_extents[dimension]; i++) {
    add_slice_to_data(
        make_not_null(&overlap_data),
        data_on_slice(volume_data, volume_extents, dimension,
                      index_to_slice_at(volume_extents, direction, i)),
        overlap_extents, dimension,
        index_to_slice_at(overlap_extents, direction, i));
  }
  return overlap_data;
}

/// Extend the overlap data to the full mesh by filling it with zeros and adding
/// the overlapping slices
template <size_t Dim, typename TagsList>
Variables<TagsList> extended_overlap_data(
    const Variables<TagsList>& overlap_data, const Index<Dim>& volume_extents,
    const Index<Dim>& overlap_extents,
    const Direction<Dim>& direction) noexcept {
  Variables<TagsList> extended_data{volume_extents.product(), 0.};
  const size_t dimension = direction.dimension();
  for (size_t i = 0; i < overlap_extents[dimension]; i++) {
    add_slice_to_data(
        make_not_null(&extended_data),
        data_on_slice(overlap_data, overlap_extents, dimension,
                      index_to_slice_at(overlap_extents, direction, i)),
        volume_extents, dimension,
        index_to_slice_at(volume_extents, direction, i));
  }
  return extended_data;
}

}  // namespace schwarz_detail
}  // namespace LinearSolver
