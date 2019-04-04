// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/HwenoModifiedSolution.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <ostream>
#include <vector>

#include "DataStructures/IndexIterator.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"     // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoGridHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// Compute the quadrature weights associated with a Mesh, by taking the tensor
// product of the 1D quadrature weights in each logical dimension of the Mesh.
template <size_t VolumeDim>
DataVector compute_quadrature_weights(const Mesh<VolumeDim>& mesh) noexcept {
  std::array<DataVector, VolumeDim> quadrature_weights_1d{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    gsl::at(quadrature_weights_1d, d) =
        Spectral::quadrature_weights(mesh.slice_through(d));
  }

  DataVector result(mesh.number_of_grid_points(), 1.);
  for (IndexIterator<VolumeDim> i(mesh.extents()); i; ++i) {
    result[i.collapsed_index()] = quadrature_weights_1d[0][i()[0]];
    for (size_t d = 1; d < VolumeDim; ++d) {
      result[i.collapsed_index()] *= gsl::at(quadrature_weights_1d, d)[i()[d]];
    }
  }
  return result;
}

// Compute the matrix that interpolates data from source_mesh to the grid points
// of target_mesh, by taking the tensor product of the 1D interpolation matrices
// in each logical dimension of the meshes.
//
// In particular, for this Hweno application:
// - source_mesh is the mesh of local element
// - target_mesh is the mesh of the neighboring element in direction. Note that
//   there can only be one neighbor in this direction, and this is checked in
//   the call to `neighbor_grid_points_in_local_logical_coordinates`.
// - rectilinear elements are assumed
template <size_t VolumeDim>
Matrix compute_interpolation_matrix(
    const Mesh<VolumeDim>& source_mesh, const Mesh<VolumeDim>& target_mesh,
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction) noexcept {
  // The grid points of source_mesh and target_mesh must be in the same
  // coordinates to construct the interpolation matrix. Here we get the points
  // of target_mesh in the local logical coordinates.
  const auto target_1d_coords =
      SlopeLimiters::Weno_detail::neighbor_grid_points_in_local_logical_coords(
          source_mesh, target_mesh, element, direction);

  const auto interpolation_matrices_1d =
      intrp::RegularGrid<VolumeDim>(source_mesh, target_mesh, target_1d_coords)
          .interpolation_matrices();

  // The 1D interpolation matrices can be empty, if there is no need to
  // interpolate in that particular direction (i.e., the interpolation in that
  // direction is identity). This function will help us undo the optimization
  // by returning elements of an identity matrix if an empty matrix is found.
  const auto matrix_element = [&interpolation_matrices_1d](
      const size_t dim, const size_t i, const size_t j) noexcept {
    const auto& matrix = gsl::at(interpolation_matrices_1d, dim);
    if (matrix.rows() * matrix.columns() == 0) {
      return (i == j) ? 1. : 0.;
    } else {
      return matrix(i, j);
    }
  };

  Matrix result(target_mesh.number_of_grid_points(),
                source_mesh.number_of_grid_points(), 0.);
  for (IndexIterator<VolumeDim> i(target_mesh.extents()); i; ++i) {
    for (IndexIterator<VolumeDim> j(source_mesh.extents()); j; ++j) {
      result(i.collapsed_index(), j.collapsed_index()) =
          matrix_element(0, i()[0], j()[0]);
      for (size_t d = 1; d < VolumeDim; ++d) {
        result(i.collapsed_index(), j.collapsed_index()) *=
            matrix_element(d, i()[d], j()[d]);
      }
    }
  }
  return result;
}

// Helper to compute v_i M_ij
// The input vector must have as many elements as the matrix has rows.
// The output vector has as many elements as the matrix has columns.
DataVector vector_dot_matrix(const DataVector& vector,
                             const Matrix& matrix) noexcept {
  const size_t number_of_rows = matrix.rows();
  const size_t number_of_columns = matrix.columns();
  ASSERT(vector.size() == number_of_rows, "Tried to multiply a vector of size "
                                              << vector.size()
                                              << " with a matrix with "
                                              << number_of_rows << " rows");
  DataVector result(number_of_columns, 0.);
  for (size_t i = 0; i < number_of_rows; ++i) {
    for (size_t j = 0; j < number_of_columns; ++j) {
      result[j] += vector[i] * matrix(i, j);
    }
  }
  return result;
}

// Compute the matrix A_jk for the constrained fit. For details, see the
// documentation of `compute_hweno_modified_neighbor_solution`.
template <size_t VolumeDim>
Matrix compute_Ajk_inverse_matrix(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const DataVector& quadrature_weights,
    const DirectionMap<VolumeDim, Matrix>& interpolation_matrices,
    const DirectionMap<VolumeDim, DataVector>&
        quadrature_weights_dot_interpolation_matrices,
    const Direction<VolumeDim>& primary_direction,
    const Direction<VolumeDim>& skipped_direction) noexcept {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  Matrix Ajk(number_of_grid_points, number_of_grid_points, 0.);

  // Loop only over directions where there is a neighbor
  const std::vector<Direction<VolumeDim>>
      directions_with_neighbors = [&element]() noexcept {
    std::vector<Direction<VolumeDim>> result;
    for (const auto& dir_and_neighbors : element.neighbors()) {
      result.push_back(dir_and_neighbors.first);
    }
    return result;
  }
  ();

  for (const auto& dir : directions_with_neighbors) {
    if (dir == skipped_direction) {
      continue;
    }

    const auto& neighbor_mesh = mesh;
    const auto& neighbor_quadrature_weights = quadrature_weights;
    const auto& interpolation_matrix = interpolation_matrices.at(dir);
    const auto& weights_dot_interpolation_matrix =
        quadrature_weights_dot_interpolation_matrices.at(dir);

    // Add terms from the primary neighbor
    if (dir == primary_direction) {
      for (size_t i = 0; i < neighbor_mesh.number_of_grid_points(); ++i) {
        for (size_t j = 0; j < number_of_grid_points; ++j) {
          for (size_t k = 0; k < number_of_grid_points; ++k) {
            Ajk(j, k) += neighbor_quadrature_weights[i] *
                         interpolation_matrix(i, j) *
                         interpolation_matrix(i, k);
          }
        }
      }
    }
    // Add terms from the other neighbors
    else {
      for (size_t j = 0; j < number_of_grid_points; ++j) {
        for (size_t k = 0; k < number_of_grid_points; ++k) {
          Ajk(j, k) += weights_dot_interpolation_matrix[j] *
                       weights_dot_interpolation_matrix[k];
        }
      }
    }
  }

  // Invert matrix in-place before returning
  blaze::invert<blaze::asSymmetric>(Ajk);
  return Ajk;
}

}  // namespace

namespace SlopeLimiters {
namespace Hweno_detail {

template <size_t VolumeDim>
HwenoConstrainedFitCache<VolumeDim>::HwenoConstrainedFitCache(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh) noexcept {
  quadrature_weights = compute_quadrature_weights(mesh);

  // Loop only over directions where there is a neighbor
  const std::vector<Direction<VolumeDim>>
      directions_with_neighbors = [&element]() noexcept {
    std::vector<Direction<VolumeDim>> result;
    for (const auto& dir_and_neighbors : element.neighbors()) {
      result.push_back(dir_and_neighbors.first);
    }
    return result;
  }
  ();

  for (const auto& primary_dir : directions_with_neighbors) {
    interpolation_matrices[primary_dir] =
        compute_interpolation_matrix(mesh, mesh, element, primary_dir);
    quadrature_weights_dot_interpolation_matrices[primary_dir] =
        vector_dot_matrix(quadrature_weights,
                          interpolation_matrices.at(primary_dir));
  }

  for (const auto& primary_dir : directions_with_neighbors) {
    for (const auto& skipped_dir : directions_with_neighbors) {
      // Skip the nonsensical case where the primary and skipped neighbors
      // are the same. This can never be true, and only arises here because
      // of how the data is organized in the cache.
      if (primary_dir != skipped_dir) {
        Ajk_inverse_matrices[primary_dir][skipped_dir] =
            compute_Ajk_inverse_matrix(
                element, mesh, quadrature_weights, interpolation_matrices,
                quadrature_weights_dot_interpolation_matrices, primary_dir,
                skipped_dir);
      }
    }
  }
}

namespace {

template <size_t VolumeDim, size_t DummyIndex>
const HwenoConstrainedFitCache<VolumeDim>& hweno_constrained_fit_cache_impl(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh) noexcept {
  // todo checks
  static const HwenoConstrainedFitCache<VolumeDim> result(element, mesh);
  return result;
}

template <size_t VolumeDim, size_t... Is>
const HwenoConstrainedFitCache<VolumeDim>& hweno_constrained_fit_cache(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    std::index_sequence<Is...> /*dummy_indices*/) noexcept {
  // todo checks
  static const std::array<
      const HwenoConstrainedFitCache<VolumeDim>& (*)(const Element<VolumeDim>&,
                                                     const Mesh<VolumeDim>&),
      sizeof...(Is)>
      cache{{&hweno_constrained_fit_cache_impl<VolumeDim, Is>...}};

  const size_t collapsed_element_info = [&element]() noexcept {
    std::bitset<2 * VolumeDim> bits;
    for (size_t dim = 0; dim < VolumeDim; ++dim) {
      for (const Side& side : {Side::Lower, Side::Upper}) {
        const Direction<VolumeDim> dir(dim, side);
        // Is there a neighbor in this direction?
        const bool neighbor_exists =
            (element.neighbors().find(dir) != element.neighbors().end());
        // Index into bitset for this direction
        const size_t index = 2 * dim + (side == Side::Lower ? 0 : 1);
        bits[index] = neighbor_exists;
      }
    }
    return bits.to_ulong();
  }
  ();
  ASSERT(collapsed_element_info >= 0 and collapsed_element_info < sizeof...(Is),
         "Got collapsed_element_info = " << collapsed_element_info
                                         << ", but expect only "
                                         << sizeof...(Is) << " configurations");
  return gsl::at(cache, collapsed_element_info)(element, mesh);
}

}  // namespace

template <size_t VolumeDim>
const HwenoConstrainedFitCache<VolumeDim>& hweno_constrained_fit_cache(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh) noexcept {
  return hweno_constrained_fit_cache<VolumeDim>(
      element, mesh, std::make_index_sequence<two_to_the(2 * VolumeDim)>{});
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                             \
  template class HwenoConstrainedFitCache<DIM(data)>;    \
  template const HwenoConstrainedFitCache<DIM(data)>&    \
  hweno_constrained_fit_cache(const Element<DIM(data)>&, \
                              const Mesh<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Hweno_detail
}  // namespace SlopeLimiters
