// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/OrientationMapHelpers.hpp"

namespace LinearSolver {
namespace schwarz_detail {

template <size_t Dim>
Index<Dim> overlap_extents(const Index<Dim>& volume_extents,
                           const size_t overlap,
                           const size_t dimension) noexcept {
  auto overlap_extents = volume_extents;
  overlap_extents[dimension] = overlap;
  return overlap_extents;
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

template <typename TagsList>
Variables<TagsList> orient_data_on_overlap(
    const Variables<TagsList>& vars) noexcept {
  return vars;
}

}  // namespace schwarz_detail
}  // namespace LinearSolver
