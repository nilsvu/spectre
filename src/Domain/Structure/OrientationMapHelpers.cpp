// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/OrientationMapHelpers.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {

// 1D data can be aligned or anti-aligned
std::vector<size_t> compute_offset_permutation(
    const Index<1>& extents, const bool neighbor_axis_is_aligned) {
  std::vector<size_t> oriented_offsets(extents.product());
  std::iota(oriented_offsets.begin(), oriented_offsets.end(), 0);
  if (not neighbor_axis_is_aligned) {
    std::reverse(oriented_offsets.begin(), oriented_offsets.end());
  }
  return oriented_offsets;
}

// 2D data can have 8 different data-storage orders relative to the neighbor.
// These are determined by whether the new data-storage order varies fastest by
// the lowest dim or the highest dim, and by whether each axis is aligned or
// anti-aligned.
std::vector<size_t> compute_offset_permutation(
    const Index<2>& extents, const bool neighbor_first_axis_is_aligned,
    const bool neighbor_second_axis_is_aligned,
    const bool neighbor_axes_are_transposed) {
  std::vector<size_t> oriented_offsets(extents.product());
  // Reduce the number of cases to explicitly write out by 4, by encoding the
  // (anti-)alignment of each axis as numerical factors ("offset" and "step")
  // that then contribute in identically-structured loops.
  // But doing this requires mixing positive and negative factors, so we cast
  // from size_t to int, do the work, then cast from int back to size_t.
  const auto num_pts_1 = static_cast<int>(extents[0]);
  const auto num_pts_2 = static_cast<int>(extents[1]);
  const int i1_offset = neighbor_first_axis_is_aligned ? 0 : num_pts_1 - 1;
  const int i1_step = neighbor_first_axis_is_aligned ? 1 : -1;
  const int i2_offset = neighbor_second_axis_is_aligned ? 0 : num_pts_2 - 1;
  const int i2_step = neighbor_second_axis_is_aligned ? 1 : -1;
  if (neighbor_axes_are_transposed) {
    for (int i2 = 0; i2 < num_pts_2; ++i2) {
      for (int i1 = 0; i1 < num_pts_1; ++i1) {
        // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
        oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2)] =
            // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
            static_cast<size_t>((i2_offset + i2_step * i2) +
                                num_pts_2 * (i1_offset + i1_step * i1));
      }
    }
  } else {
    for (int i2 = 0; i2 < num_pts_2; ++i2) {
      for (int i1 = 0; i1 < num_pts_1; ++i1) {
        // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
        oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2)] =
            // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
            static_cast<size_t>((i1_offset + i1_step * i1) +
                                num_pts_1 * (i2_offset + i2_step * i2));
      }
    }
  }
  return oriented_offsets;
}

// 3D data can have 48 (!) different data-storage orders relative to the
// neighbor. A factor of 6 arises from the different ways in which the three
// dimensions can be ordered from fastest to slowest varying. The remaining
// factor of 8 arises from having two possible directions (aligned or
// anti-aligned) for each of the three axes.
std::vector<size_t> compute_offset_permutation(
    const Index<3>& extents, const bool neighbor_first_axis_is_aligned,
    const bool neighbor_second_axis_is_aligned,
    const bool neighbor_third_axis_is_aligned,
    const std::array<size_t, 3>& neighbor_axis_permutation) {
  std::vector<size_t> oriented_offsets(extents.product());
  // Reduce the number of cases to explicitly write out by 8, by encoding the
  // (anti-)alignment of each axis as numerical factors ("offset" and "step")
  // that then contribute in identically-structured loops.
  // But doing this requires mixing positive and negative factors, so we cast
  // from size_t to int, do the work, then cast from int back to size_t.
  const auto num_pts_1 = static_cast<int>(extents[0]);
  const auto num_pts_2 = static_cast<int>(extents[1]);
  const auto num_pts_3 = static_cast<int>(extents[2]);
  const int i1_offset = neighbor_first_axis_is_aligned ? 0 : num_pts_1 - 1;
  const int i1_step = neighbor_first_axis_is_aligned ? 1 : -1;
  const int i2_offset = neighbor_second_axis_is_aligned ? 0 : num_pts_2 - 1;
  const int i2_step = neighbor_second_axis_is_aligned ? 1 : -1;
  const int i3_offset = neighbor_third_axis_is_aligned ? 0 : num_pts_3 - 1;
  const int i3_step = neighbor_third_axis_is_aligned ? 1 : -1;
  // The three cyclic permutations of the dimensions 0, 1, 2
  // Note that these do not necessarily lead to right-handed coordinate
  // systems, because the final "handedness" also depends on whether
  // each axis is aligned or anti-aligned.
  if (neighbor_axis_permutation == make_array(0_st, 1_st, 2_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              static_cast<size_t>((i1_offset + i1_step * i1) +
                                  num_pts_1 * (i2_offset + i2_step * i2) +
                                  num_pts_1 * num_pts_2 *
                                      (i3_offset + i3_step * i3));
        }
      }
    }
  } else if (neighbor_axis_permutation == make_array(1_st, 2_st, 0_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              static_cast<size_t>((i3_offset + i3_step * i3) +
                                  num_pts_3 * (i1_offset + i1_step * i1) +
                                  num_pts_3 * num_pts_1 *
                                      (i2_offset + i2_step * i2));
        }
      }
    }
  } else if (neighbor_axis_permutation == make_array(2_st, 0_st, 1_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              static_cast<size_t>((i2_offset + i2_step * i2) +
                                  num_pts_2 * (i3_offset + i3_step * i3) +
                                  num_pts_2 * num_pts_3 *
                                      (i1_offset + i1_step * i1));
        }
      }
    }
  }
  // The three acyclic permutations
  else if (neighbor_axis_permutation == make_array(0_st, 2_st, 1_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              static_cast<size_t>((i1_offset + i1_step * i1) +
                                  num_pts_1 * (i3_offset + i3_step * i3) +
                                  num_pts_1 * num_pts_3 *
                                      (i2_offset + i2_step * i2));
        }
      }
    }
  } else if (neighbor_axis_permutation == make_array(2_st, 1_st, 0_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              static_cast<size_t>((i3_offset + i3_step * i3) +
                                  num_pts_3 * (i2_offset + i2_step * i2) +
                                  num_pts_3 * num_pts_2 *
                                      (i1_offset + i1_step * i1));
        }
      }
    }
  } else {  // make_array(1_st, 0_st, 2_st)
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              static_cast<size_t>((i2_offset + i2_step * i2) +
                                  num_pts_2 * (i1_offset + i1_step * i1) +
                                  num_pts_2 * num_pts_1 *
                                      (i3_offset + i3_step * i3));
        }
      }
    }
  }
  return oriented_offsets;
}

std::vector<size_t> oriented_offset(
    const Index<1>& extents, const OrientationMap<1>& orientation_of_neighbor) {
  const Direction<1> neighbor_axis =
      orientation_of_neighbor(Direction<1>::upper_xi());
  const bool is_aligned = (neighbor_axis.side() == Side::Upper);
  return compute_offset_permutation(extents, is_aligned);
}

std::vector<size_t> oriented_offset(
    const Index<2>& extents, const OrientationMap<2>& orientation_of_neighbor) {
  const Direction<2> neighbor_first_axis =
      orientation_of_neighbor(Direction<2>::upper_xi());
  const Direction<2> neighbor_second_axis =
      orientation_of_neighbor(Direction<2>::upper_eta());
  const bool axes_are_transposed =
      (neighbor_first_axis.dimension() > neighbor_second_axis.dimension());
  const bool neighbor_first_axis_is_aligned =
      (Side::Upper == neighbor_first_axis.side());
  const bool neighbor_second_axis_is_aligned =
      (Side::Upper == neighbor_second_axis.side());

  return compute_offset_permutation(extents, neighbor_first_axis_is_aligned,
                                    neighbor_second_axis_is_aligned,
                                    axes_are_transposed);
}

std::vector<size_t> oriented_offset(
    const Index<3>& extents, const OrientationMap<3>& orientation_of_neighbor) {
  const Direction<3> neighbor_first_axis =
      orientation_of_neighbor(Direction<3>::upper_xi());
  const Direction<3> neighbor_second_axis =
      orientation_of_neighbor(Direction<3>::upper_eta());
  const Direction<3> neighbor_third_axis =
      orientation_of_neighbor(Direction<3>::upper_zeta());

  const bool neighbor_first_axis_is_aligned =
      (Side::Upper == neighbor_first_axis.side());
  const bool neighbor_second_axis_is_aligned =
      (Side::Upper == neighbor_second_axis.side());
  const bool neighbor_third_axis_is_aligned =
      (Side::Upper == neighbor_third_axis.side());

  const auto neighbor_axis_permutation = make_array(
      neighbor_first_axis.dimension(), neighbor_second_axis.dimension(),
      neighbor_third_axis.dimension());

  return compute_offset_permutation(
      extents, neighbor_first_axis_is_aligned, neighbor_second_axis_is_aligned,
      neighbor_third_axis_is_aligned, neighbor_axis_permutation);
}

std::vector<size_t> oriented_offset_on_slice(
    const Index<0>& /*slice_extents*/, const size_t /*sliced_dim*/,
    const OrientationMap<1>& /*orientation_of_neighbor*/) {
  // There is only one point on a slice of a 1D mesh
  return {0};
}

std::vector<size_t> oriented_offset_on_slice(
    const Index<1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<2>& orientation_of_neighbor) {
  const Direction<2> my_slice_axis =
      (0 == sliced_dim ? Direction<2>::upper_eta() : Direction<2>::upper_xi());
  const Direction<2> neighbor_slice_axis =
      orientation_of_neighbor(my_slice_axis);
  const bool is_aligned = (neighbor_slice_axis.side() == Side::Upper);
  return compute_offset_permutation(slice_extents, is_aligned);
}

std::vector<size_t> oriented_offset_on_slice(
    const Index<2>& slice_extents, const size_t sliced_dim,
    const OrientationMap<3>& orientation_of_neighbor) {
  const std::array<size_t, 2> dims_of_slice =
      (0 == sliced_dim ? make_array(1_st, 2_st)
                       : (1 == sliced_dim) ? make_array(0_st, 2_st)
                                           : make_array(0_st, 1_st));
  const bool neighbor_axes_are_transposed =
      (orientation_of_neighbor(dims_of_slice[0]) >
       orientation_of_neighbor(dims_of_slice[1]));
  const Direction<3> neighbor_first_axis =
      orientation_of_neighbor(Direction<3>(dims_of_slice[0], Side::Upper));
  const Direction<3> neighbor_second_axis =
      orientation_of_neighbor(Direction<3>(dims_of_slice[1], Side::Upper));
  const bool neighbor_first_axis_is_aligned =
      (Side::Upper == neighbor_first_axis.side());
  const bool neighbor_second_axis_is_aligned =
      (Side::Upper == neighbor_second_axis.side());

  return compute_offset_permutation(
      slice_extents, neighbor_first_axis_is_aligned,
      neighbor_second_axis_is_aligned, neighbor_axes_are_transposed);
}

template <typename T>
void orient_each_component(
    const gsl::not_null<gsl::span<T>*> oriented_variables,
    const gsl::span<const T>& variables, const size_t num_pts,
    const std::vector<size_t>& oriented_offset) {
  const size_t num_components = variables.size() / num_pts;
  ASSERT(oriented_variables->size() == variables.size(),
         "The number of oriented variables, "
             << oriented_variables->size() / num_pts
             << ", must be equal to the number of variables, "
             << variables.size() / num_pts);
  for (size_t component_index = 0; component_index < num_components;
       ++component_index) {
    const size_t offset = component_index * num_pts;
    for (size_t s = 0; s < num_pts; ++s) {
      gsl::at((*oriented_variables), offset + oriented_offset[s]) =
          gsl::at(variables, offset + s);
    }
  }
}
}  // namespace

template <size_t VolumeDim>
Mesh<VolumeDim - 1> orient_mesh_on_slice(
    const Mesh<VolumeDim - 1>& mesh_on_slice, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  if constexpr (VolumeDim < 3) {
    return mesh_on_slice;
  } else {
    if (orientation_of_neighbor.is_aligned()) {
      return mesh_on_slice;
    }
    const size_t first_dim_of_slice = sliced_dim == 0 ? 1 : 0;
    const size_t second_dim_of_slice = sliced_dim == 2 ? 1 : 2;
    if (orientation_of_neighbor(first_dim_of_slice) >
        orientation_of_neighbor(second_dim_of_slice)) {
      return Mesh<2>{
          {{mesh_on_slice.extents(1), mesh_on_slice.extents(0)}},
          {{mesh_on_slice.basis(1), mesh_on_slice.basis(0)}},
          {{mesh_on_slice.quadrature(1), mesh_on_slice.quadrature(0)}}};
    } else {
      return mesh_on_slice;
    }
  }
}

template <typename VectorType, size_t VolumeDim>
void orient_variables(
    const gsl::not_null<VectorType*> result, const VectorType& variables,
    const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  ASSERT(result->size() == variables.size(),
         "Result should have size " << variables.size() << " but has size "
                                    << result->size());
  const size_t number_of_grid_points = extents.product();
  ASSERT(variables.size() % number_of_grid_points == 0,
         "The size of the variables must be divisible by the number of grid "
         "points. Number of grid points: "
             << number_of_grid_points << " size: " << variables.size());
  // Skip work (aside from a copy) if neighbor is aligned
  if (orientation_of_neighbor.is_aligned()) {
    (*result) = variables;
    return;
  }

  const auto oriented_extents =
      oriented_offset(extents, orientation_of_neighbor);
  auto oriented_vars_view = gsl::make_span(result->data(), result->size());
  orient_each_component(make_not_null(&oriented_vars_view),
                        gsl::make_span(variables.data(), variables.size()),
                        number_of_grid_points, oriented_extents);
}

template <typename VectorType, size_t VolumeDim>
void orient_variables_on_slice(
    const gsl::not_null<VectorType*> result,
    const VectorType& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  ASSERT(result->size() == variables_on_slice.size(),
         "Result should have size " << variables_on_slice.size()
                                    << " but has size " << result->size());
  const size_t number_of_grid_points = slice_extents.product();
  ASSERT(variables_on_slice.size() % number_of_grid_points == 0,
         "The size of the variables must be divisible by the number of grid "
         "points. Number of grid points: "
             << number_of_grid_points
             << " size: " << variables_on_slice.size());
  // Skip work (aside from a copy) if neighbor slice is aligned
  if (orientation_of_neighbor.is_aligned()) {
    (*result) = variables_on_slice;
    return;
  }

  const auto oriented_offset = oriented_offset_on_slice(
      slice_extents, sliced_dim, orientation_of_neighbor);

  auto oriented_vars_view = gsl::make_span(result->data(), result->size());
  orient_each_component(
      make_not_null(&oriented_vars_view),
      gsl::make_span(variables_on_slice.data(), variables_on_slice.size()),
      number_of_grid_points, oriented_offset);
}

template <typename ValueType, size_t VolumeDim>
std::vector<ValueType> orient_variables(
    const std::vector<ValueType>& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  std::vector<ValueType> oriented_variables(variables.size());
  using VectorType =
      std::conditional_t<std::is_same_v<ValueType, std::complex<double>>,
                         ComplexDataVector, DataVector>;
  VectorType result(oriented_variables.data(), oriented_variables.size());
  orient_variables(
      make_not_null(&result),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      VectorType(const_cast<ValueType*>(variables.data()), variables.size()),
      extents, orientation_of_neighbor);
  return oriented_variables;
}

template <typename VectorType, size_t VolumeDim>
VectorType orient_variables(
    const VectorType& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  VectorType oriented_variables{variables.size()};
  orient_variables(make_not_null(&oriented_variables), variables, extents,
                   orientation_of_neighbor);
  return oriented_variables;
}

template <typename ValueType, size_t VolumeDim>
std::vector<ValueType> orient_variables_on_slice(
    const std::vector<ValueType>& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  std::vector<ValueType> oriented_variables(variables_on_slice.size());
  using VectorType =
      std::conditional_t<std::is_same_v<ValueType, std::complex<double>>,
                         ComplexDataVector, DataVector>;
  VectorType result(oriented_variables.data(), oriented_variables.size());
  orient_variables_on_slice(
      make_not_null(&result),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      VectorType(const_cast<ValueType*>(variables_on_slice.data()),
                 variables_on_slice.size()),
      slice_extents, sliced_dim, orientation_of_neighbor);
  return oriented_variables;
}

template <typename VectorType, size_t VolumeDim>
VectorType orient_variables_on_slice(
    const VectorType& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  VectorType oriented_variables{variables_on_slice.size()};
  orient_variables_on_slice(make_not_null(&oriented_variables),
                            variables_on_slice, slice_extents, sliced_dim,
                            orientation_of_neighbor);
  return oriented_variables;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

template Mesh<0> orient_mesh_on_slice(
    const Mesh<0>& mesh_on_slice, const size_t sliced_dim,
    const OrientationMap<1>& orientation_of_neighbor);
template Mesh<1> orient_mesh_on_slice(
    const Mesh<1>& mesh_on_slice, const size_t sliced_dim,
    const OrientationMap<2>& orientation_of_neighbor);
template Mesh<2> orient_mesh_on_slice(
    const Mesh<2>& mesh_on_slice, const size_t sliced_dim,
    const OrientationMap<3>& orientation_of_neighbor);

#define INSTANTIATION(r, data)                                                 \
  template void orient_variables(                                              \
      const gsl::not_null<DTYPE(data)*> result, const DTYPE(data) & variables, \
      const Index<DIM(data)>& extents,                                         \
      const OrientationMap<DIM(data)>& orientation_of_neighbor);               \
  template void orient_variables_on_slice(                                     \
      const gsl::not_null<DTYPE(data)*> result,                                \
      const DTYPE(data) & variables_on_slice,                                  \
      const Index<DIM(data) - 1>& slice_extents, size_t sliced_dim,            \
      const OrientationMap<DIM(data)>& orientation_of_neighbor);               \
  template std::vector<DTYPE(data)::value_type> orient_variables(              \
      const std::vector<DTYPE(data)::value_type>& variables,                   \
      const Index<DIM(data)>& extents,                                         \
      const OrientationMap<DIM(data)>& orientation_of_neighbor);               \
  template DTYPE(data) orient_variables(                                       \
      const DTYPE(data) & variables, const Index<DIM(data)>& extents,          \
      const OrientationMap<DIM(data)>& orientation_of_neighbor);               \
  template std::vector<DTYPE(data)::value_type> orient_variables_on_slice(     \
      const std::vector<DTYPE(data)::value_type>& variables,                   \
      const Index<DIM(data) - 1>& extents, size_t sliced_dim,                  \
      const OrientationMap<DIM(data)>& orientation_of_neighbor);               \
  template DTYPE(data) orient_variables_on_slice(                              \
      const DTYPE(data) & variables, const Index<DIM(data) - 1>& extents,      \
      size_t sliced_dim,                                                       \
      const OrientationMap<DIM(data)>& orientation_of_neighbor);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, ComplexDataVector),
                        (1, 2, 3))

#undef INSTANTIATION
#undef DIM
