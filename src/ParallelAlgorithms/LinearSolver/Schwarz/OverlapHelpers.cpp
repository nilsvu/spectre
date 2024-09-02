// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>  // std::accumulate
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace LinearSolver::Schwarz {

size_t overlap_extent(const size_t volume_extent, const size_t max_overlap) {
  return std::min(max_overlap, volume_extent - 1);
}

namespace {
template <size_t Dim>
void assert_overlap_extent(const Index<Dim>& volume_extents,
                           const size_t overlap_extent,
                           const size_t overlap_dimension) {
  ASSERT(overlap_dimension < Dim,
         "Invalid dimension '" << overlap_dimension << "' in " << Dim << "D.");
  ASSERT(overlap_extent <= volume_extents[overlap_dimension],
         "Overlap extent '" << overlap_extent << "' exceeds volume extents '"
                            << volume_extents << "' in overlap dimension '"
                            << overlap_dimension << "'.");
}
}  // namespace

template <size_t Dim>
size_t overlap_num_points(const Index<Dim>& volume_extents,
                          const size_t overlap_extent,
                          const size_t overlap_dimension) {
  assert_overlap_extent(volume_extents, overlap_extent, overlap_dimension);
  return volume_extents.slice_away(overlap_dimension).product() *
         overlap_extent;
}

double overlap_width(const size_t overlap_extent,
                     const DataVector& collocation_points) {
  ASSERT(overlap_extent < collocation_points.size(),
         "Overlap extent is "
             << overlap_extent
             << " but must be strictly smaller than the number of grid points ("
             << collocation_points.size() << ") to compute an overlap width.");
  for (size_t i = 0; i < collocation_points.size() / 2; ++i) {
    ASSERT(equal_within_roundoff(
               -collocation_points[i],
               collocation_points[collocation_points.size() - i - 1]),
           "Assuming the 'collocation_points' are symmetric, but they are not: "
               << collocation_points);
  }
  // The overlap boundary index lies one point outside the region covered by the
  // overlap coordinates (see `LinearSolver::Schwarz::overlap_extent`).
  return collocation_points[overlap_extent] - collocation_points[0];
}

template <size_t Dim>
OverlapIterator::OverlapIterator(const Index<Dim>& volume_extents,
                                 const size_t overlap_extent,
                                 const Direction<Dim>& direction)
    : size_{overlap_num_points(volume_extents, overlap_extent,
                               direction.dimension())},
      num_slices_{overlap_extent},
      stride_{std::accumulate(volume_extents.begin(),
                              volume_extents.begin() + direction.dimension(),
                              1_st, std::multiplies<size_t>())},
      stride_count_{0},
      jump_{(volume_extents[direction.dimension()] - num_slices_) * stride_},
      initial_offset_{stride_ * (direction.side() == Side::Lower
                                     ? 0
                                     : (volume_extents[direction.dimension()] -
                                        num_slices_))},
      volume_offset_{initial_offset_},
      overlap_offset_{0} {
  assert_overlap_extent(volume_extents, overlap_extent, direction.dimension());
}

OverlapIterator::operator bool() const { return overlap_offset_ < size_; }

OverlapIterator& OverlapIterator::operator++() {
  ++volume_offset_;
  ++overlap_offset_;
  ++stride_count_;
  if (stride_count_ == stride_ * num_slices_) {
    volume_offset_ += jump_;
    stride_count_ = 0;
  }
  return *this;
}

size_t OverlapIterator::volume_offset() const { return volume_offset_; }

size_t OverlapIterator::overlap_offset() const { return overlap_offset_; }

void OverlapIterator::reset() {
  volume_offset_ = initial_offset_;
  overlap_offset_ = 0;
  stride_count_ = 0;
}

namespace detail {

template <typename ValueType, size_t Dim>
void data_on_overlap_impl(ValueType* overlap_data, const ValueType* volume_data,
                          const size_t num_components,
                          const Index<Dim>& volume_extents,
                          const size_t overlap_extent,
                          const Direction<Dim>& direction) {
  const size_t volume_num_points = volume_extents.product();
  const size_t overlap_num_points = LinearSolver::Schwarz::overlap_num_points(
      volume_extents, overlap_extent, direction.dimension());
  for (OverlapIterator overlap_iterator{volume_extents, overlap_extent,
                                        direction};
       overlap_iterator; ++overlap_iterator) {
    for (size_t j = 0; j < num_components; ++j) {
      const size_t volume_data_index =
          overlap_iterator.volume_offset() + j * volume_num_points;
      const size_t overlap_data_index =
          overlap_iterator.overlap_offset() + j * overlap_num_points;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      overlap_data[overlap_data_index] = volume_data[volume_data_index];
    }
  }
}

template <typename ValueType, size_t Dim>
void add_overlap_data_impl(ValueType* volume_data,
                           const ValueType* overlap_data,
                           const size_t num_components,
                           const Index<Dim>& volume_extents,
                           const size_t overlap_extent,
                           const Direction<Dim>& direction) {
  const size_t volume_num_points = volume_extents.product();
  const size_t overlap_num_points = LinearSolver::Schwarz::overlap_num_points(
      volume_extents, overlap_extent, direction.dimension());
  for (OverlapIterator overlap_iterator{volume_extents, overlap_extent,
                                        direction};
       overlap_iterator; ++overlap_iterator) {
    for (size_t j = 0; j < num_components; ++j) {
      const size_t volume_data_index =
          overlap_iterator.volume_offset() + j * volume_num_points;
      const size_t overlap_data_index =
          overlap_iterator.overlap_offset() + j * overlap_num_points;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      volume_data[volume_data_index] += overlap_data[overlap_data_index];
    }
  }
}

}  // namespace detail

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(r, data)                                                   \
  template size_t overlap_num_points(const Index<DIM(data)>&, size_t, size_t); \
  template OverlapIterator::OverlapIterator(const Index<DIM(data)>&, size_t,   \
                                            const Direction<DIM(data)>&);
#define INSTANTIATE_DTYPE(r, data)                                       \
  template void detail::add_overlap_data_impl(                           \
      DTYPE(data)*, const DTYPE(data)*, size_t, const Index<DIM(data)>&, \
      size_t, const Direction<DIM(data)>&);                              \
  template void detail::data_on_overlap_impl(                            \
      DTYPE(data)*, const DTYPE(data)*, size_t, const Index<DIM(data)>&, \
      size_t, const Direction<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE, (1, 2, 3),
                        (double, std::complex<double>))

#undef DIM
#undef INSTANTIATE

}  // namespace LinearSolver::Schwarz
