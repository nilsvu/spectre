// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Multigrid/InterMeshOperators.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace LinearSolver::multigrid {

namespace {

// Find the logical coordinates of the fine-mesh collocation points in the
// coarse (parent) mesh
DataVector fine_points_in_coarse_mesh(
    const Mesh<1>& fine_mesh, const SegmentId& child_segment_id,
    const SegmentId& parent_segment_id) noexcept {
  if (child_segment_id == parent_segment_id) {
    return Spectral::collocation_points(fine_mesh);
  } else {
    ASSERT(child_segment_id.id_of_parent() == parent_segment_id,
           "Segment id '" << parent_segment_id << "' is not the parent of '"
                          << child_segment_id << "'.");
    const double sign =
        parent_segment_id.id_of_child(Side::Lower) == child_segment_id ? -1.
                                                                       : 1.;
    return 0.5 * Spectral::collocation_points(fine_mesh) + sign * 0.5;
  }
}

}  // namespace

template <bool MassiveOperator, size_t Dim>
std::array<Matrix, Dim> restriction_operator(
    const Mesh<Dim>& fine_mesh, const Mesh<Dim>& coarse_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids) noexcept {
  std::array<Matrix, Dim> restriction_operator{};
  for (size_t d = 0; d < Dim; d++) {
    if (child_segment_ids[d] == parent_segment_ids[d] and
        fine_mesh.extents(d) == coarse_mesh.extents(d)) {
      continue;
    }
    if constexpr (MassiveOperator) {
      restriction_operator[d] = blaze::trans(Spectral::interpolation_matrix(
          coarse_mesh.slice_through(d),
          fine_points_in_coarse_mesh(fine_mesh.slice_through(d),
                                     child_segment_ids[d],
                                     parent_segment_ids[d])));
    } else {
      // M_coarse^-1 * I_coarse_to_fine^T * M_fine
      // TODO: check if this works for massless operator if jacobians are added
      restriction_operator[d] =
          blaze::inv(Spectral::mass_matrix(coarse_mesh.slice_through(d))) *
          blaze::trans(Spectral::interpolation_matrix(
              coarse_mesh.slice_through(d),
              fine_points_in_coarse_mesh(fine_mesh.slice_through(d),
                                         child_segment_ids[d],
                                         parent_segment_ids[d]))) *
          Spectral::mass_matrix(fine_mesh.slice_through(d));
    }
  }
  return restriction_operator;
}

template <size_t Dim>
std::array<Matrix, Dim> prolongation_operator(
    const Mesh<Dim>& coarse_mesh, const Mesh<Dim>& fine_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids) noexcept {
  std::array<Matrix, Dim> prolongation_operator{};
  for (size_t d = 0; d < Dim; d++) {
    if (child_segment_ids[d] == parent_segment_ids[d] and
        fine_mesh.extents(d) == coarse_mesh.extents(d)) {
      continue;
    }
    prolongation_operator[d] = Spectral::interpolation_matrix(
        coarse_mesh.slice_through(d),
        fine_points_in_coarse_mesh(fine_mesh.slice_through(d),
                                   child_segment_ids[d],
                                   parent_segment_ids[d]));
  }
  return prolongation_operator;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                                \
  template std::array<Matrix, DIM(data)> restriction_operator<true>(        \
      const Mesh<DIM(data)>& fine_mesh, const Mesh<DIM(data)>& coarse_mesh, \
      const std::array<SegmentId, DIM(data)>& child_segment_ids,            \
      const std::array<SegmentId, DIM(data)>& parent_segment_ids) noexcept; \
  template std::array<Matrix, DIM(data)> restriction_operator<false>(       \
      const Mesh<DIM(data)>& fine_mesh, const Mesh<DIM(data)>& coarse_mesh, \
      const std::array<SegmentId, DIM(data)>& child_segment_ids,            \
      const std::array<SegmentId, DIM(data)>& parent_segment_ids) noexcept; \
  template std::array<Matrix, DIM(data)> prolongation_operator(             \
      const Mesh<DIM(data)>& fine_mesh, const Mesh<DIM(data)>& coarse_mesh, \
      const std::array<SegmentId, DIM(data)>& child_segment_ids,            \
      const std::array<SegmentId, DIM(data)>& parent_segment_ids) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace LinearSolver::multigrid
