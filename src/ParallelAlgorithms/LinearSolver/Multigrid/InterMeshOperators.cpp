// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Multigrid/InterMeshOperators.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace LinearSolver::multigrid {

namespace {

Spectral::MortarSize mortar_size(const SegmentId& child_segment_id,
                                 const SegmentId& parent_segment_id) noexcept {
  if (child_segment_id == parent_segment_id) {
    return Spectral::MortarSize::Full;
  } else {
    ASSERT(child_segment_id.id_of_parent() == parent_segment_id,
           "Segment id '" << parent_segment_id << "' is not the parent of '"
                          << child_segment_id << "'.");
    return parent_segment_id.id_of_child(Side::Lower) == child_segment_id
               ? Spectral::MortarSize::LowerHalf
               : Spectral::MortarSize::UpperHalf;
  }
}

}  // namespace

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim> restriction_operator(
    const Mesh<Dim>& fine_mesh, const Mesh<Dim>& coarse_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids,
    const bool massive) noexcept {
  static const Matrix identity{};
  auto restriction_operator = make_array<Dim>(std::cref(identity));
  for (size_t d = 0; d < Dim; d++) {
    const auto fine_mesh_d = fine_mesh.slice_through(d);
    const auto coarse_mesh_d = coarse_mesh.slice_through(d);
    if (child_segment_ids.at(d) != parent_segment_ids.at(d) or
        fine_mesh_d != coarse_mesh_d) {
      if (massive) {
        // No need to multiply with mass matrices, since they are absorbed in
        // the operand. The restriction operator is just the transpose of the
        // prolongation operator, i.e. an interpolation matrix transpose
        restriction_operator.at(d) =
            Spectral::projection_matrix_mortar_to_element_massive(
                mortar_size(child_segment_ids.at(d), parent_segment_ids.at(d)),
                coarse_mesh_d, fine_mesh_d);
      } else {
        restriction_operator.at(d) =
            Spectral::projection_matrix_mortar_to_element(
                mortar_size(child_segment_ids.at(d), parent_segment_ids.at(d)),
                coarse_mesh_d, fine_mesh_d);
      }
    }
  }
  return restriction_operator;
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim> prolongation_operator(
    const Mesh<Dim>& coarse_mesh, const Mesh<Dim>& fine_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids) noexcept {
  static const Matrix identity{};
  auto prolongation_operator = make_array<Dim>(std::cref(identity));
  for (size_t d = 0; d < Dim; d++) {
    const auto fine_mesh_d = fine_mesh.slice_through(d);
    const auto coarse_mesh_d = coarse_mesh.slice_through(d);
    if (child_segment_ids.at(d) != parent_segment_ids.at(d) or
        fine_mesh_d != coarse_mesh_d) {
      prolongation_operator.at(d) =
          Spectral::projection_matrix_element_to_mortar(
              mortar_size(child_segment_ids.at(d), parent_segment_ids.at(d)),
              fine_mesh_d, coarse_mesh_d);
    }
  }
  return prolongation_operator;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                                \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>      \
  restriction_operator(                                                     \
      const Mesh<DIM(data)>& fine_mesh, const Mesh<DIM(data)>& coarse_mesh, \
      const std::array<SegmentId, DIM(data)>& child_segment_ids,            \
      const std::array<SegmentId, DIM(data)>& parent_segment_ids,           \
      bool massive) noexcept;                                               \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>      \
  prolongation_operator(                                                    \
      const Mesh<DIM(data)>& fine_mesh, const Mesh<DIM(data)>& coarse_mesh, \
      const std::array<SegmentId, DIM(data)>& child_segment_ids,            \
      const std::array<SegmentId, DIM(data)>& parent_segment_ids) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace LinearSolver::multigrid
