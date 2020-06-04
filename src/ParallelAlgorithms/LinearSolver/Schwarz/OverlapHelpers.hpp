// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <boost/range/combine.hpp>
#include <cmath>
#include <iterator>
#include <tuple>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "Domain/Side.hpp"
#include "Utilities/ContainerHelpers.hpp"

namespace LinearSolver::Schwarz {

template <size_t Dim>
using OverlapId = std::pair<Direction<Dim>, ElementId<Dim>>;

template <size_t Dim, typename ValueType>
using OverlapMap =
    FixedHashMap<maximum_number_of_neighbors(Dim), OverlapId<Dim>, ValueType,
                 boost::hash<OverlapId<Dim>>>;

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

/*!
 * \brief Width of the overlap in the coordinates that the `full_coords` are
 * given in
 *
 * \see `LinearSolver::Schwarz::overlap_extents`
 */
double overlap_width(const DataVector& full_coords, const size_t overlap_extent,
                     const Side& side) noexcept {
  // The `full_coords` coords are typically the logical coordinates, which are
  // typically symmetric, but to be safe we take the direction into account.
  const size_t boundary_index =
      side == Side::Lower ? 0 : (full_coords.size() - 1);
  // The overlap boundary index lies one point outside the region covered by the
  // overlap coordinates (see `overlap_extent`).
  const size_t overlap_boundary_index =
      side == Side::Lower ? overlap_extent
                          : (full_coords.size() - 1 - overlap_extent);
  return std::abs(full_coords[overlap_boundary_index] -
                  full_coords[boundary_index]);
}

template <typename DataType>
auto smoothstep(const DataType& arg) noexcept {
  static const std::vector<double> coeffs{0., 15., 0., -10., 0., 3.};
  auto result = make_with_value<DataType>(arg, 0.);
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
auto extruding_weight(const DataType& logical_coords, const double width,
                      const Side& side) noexcept {
  const double sign = side == Side::Lower ? -1. : 1.;
  return blaze::evaluate(
      0.5 * (1. - sign * smoothstep(blaze::evaluate((logical_coords - sign) /
                                                    width))));
}

template <typename DataType>
auto intruding_weight(const DataType& logical_coords, const double width,
                      const Side& side) noexcept {
  const double sign = side == Side::Lower ? -1. : 1.;
  return extruding_weight(logical_coords - sign * 2., width, opposite(side));
}

/// The part of `tensor` that lies within the overlap region
template <size_t Dim, typename DataType, typename... TensorStructure>
Tensor<DataType, TensorStructure...> restrict_to_overlap(
    const Tensor<DataType, TensorStructure...>& tensor,
    const Index<Dim>& volume_extents, const Index<Dim>& overlap_extents,
    const Direction<Dim>& direction) noexcept {
  Tensor<DataType, TensorStructure...> restricted_tensor{
      overlap_extents.product()};
  const size_t dimension = direction.dimension();
  const size_t overlap = overlap_extents[dimension];
  for (size_t i = 0; i < overlap; i++) {
    SliceIterator slice_in_overlap{
        overlap_extents, dimension,
        index_to_slice_at(overlap_extents, direction, i)};
    for (SliceIterator slice_in_volume{
             volume_extents, dimension,
             index_to_slice_at(volume_extents, direction, i)};
         slice_in_volume; ++slice_in_volume) {
      for (decltype(auto) overlap_and_volume_tensor_components :
           boost::combine(restricted_tensor, tensor)) {
        boost::get<0>(
            overlap_and_volume_tensor_components)[slice_in_overlap
                                                      .volume_offset()] =
            boost::get<1>(
                overlap_and_volume_tensor_components)[slice_in_volume
                                                          .volume_offset()];
      }
      ++slice_in_overlap;
    }
  }
  return restricted_tensor;
}

/// The part of the `volume_data` that lies within the overlap region
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

/// Add the `overlap_data` to the `volume_data`
template <size_t Dim, typename VolumeTagsList, typename OverlapTagsList>
void add_overlap_data(
    const gsl::not_null<Variables<VolumeTagsList>*> volume_data,
    const Variables<OverlapTagsList>& overlap_data,
    const Index<Dim>& volume_extents, const Index<Dim>& overlap_extents,
    const Direction<Dim>& direction) noexcept {
  const size_t dimension = direction.dimension();
  for (size_t i = 0; i < overlap_extents[dimension]; i++) {
    add_slice_to_data(
        volume_data,
        data_on_slice(overlap_data, overlap_extents, dimension,
                      index_to_slice_at(overlap_extents, direction, i)),
        volume_extents, dimension,
        index_to_slice_at(volume_extents, direction, i));
  }
}

/// Extend the overlap data to the full mesh by filling it with zeros and adding
/// the overlapping slices
template <size_t Dim, typename TagsList>
Variables<TagsList> extended_overlap_data(
    const Variables<TagsList>& overlap_data, const Index<Dim>& volume_extents,
    const Index<Dim>& overlap_extents,
    const Direction<Dim>& direction) noexcept {
  Variables<TagsList> extended_data{volume_extents.product(), 0.};
  add_overlap_data(make_not_null(&extended_data), overlap_data, volume_extents,
                   overlap_extents, direction);
  return extended_data;
}

}  // namespace LinearSolver::Schwarz
