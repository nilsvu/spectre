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

// We restrict fields to the coarser (parent) mesh with a Galerkin
// projection R = M_coarse^-1 * I_coarse_to_fine^T * M_fine.
// See https://arxiv.org/pdf/1808.05320.pdf Eq 3.2.
// This is the L2 projection also used for AMR.
// Note that only "massless" fields restrict with `R`, i.e. polynomial
// approximations on the grid, and not "massive" quantities, i.e. DG residuals
// that involve an integral over basis functions and thus an application of the
// mass matrix. The latter can be restricted by simply applying the
// interpolation matrix transpose I_coarse_to_fine^T (see
// https://arxiv.org/pdf/1808.05320.pdf section 3.5).
template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim> restriction_operator(
    const Mesh<Dim>& fine_mesh, const Mesh<Dim>& coarse_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids,
    bool massive) noexcept;

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim> prolongation_operator(
    const Mesh<Dim>& coarse_mesh, const Mesh<Dim>& fine_mesh,
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids) noexcept;

}  // namespace LinearSolver::multigrid
