// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Matrix.hpp"

/// \cond
template <size_t Dim>
struct Mesh;
struct SegmentId;
/// \endcond

namespace LinearSolver::multigrid {

// We restrict the residual to the coarser (parent) mesh with a Galerkin
// projection R = M_coarse^-1 * I_coarse_to_fine^T * M_fine
// See https://arxiv.org/pdf/1808.05320.pdf Eq 3.2
// This is the L2 projection also used for AMR.
template <bool MassiveOperator, size_t Dim>
std::array<Matrix, Dim> restriction_operator(
    const Mesh<Dim>& fine_mesh, const Mesh<Dim>& coarse_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids) noexcept;

template <size_t Dim>
std::array<Matrix, Dim> prolongation_operator(
    const Mesh<Dim>& coarse_mesh, const Mesh<Dim>& fine_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids) noexcept;

}  // namespace LinearSolver::multigrid
