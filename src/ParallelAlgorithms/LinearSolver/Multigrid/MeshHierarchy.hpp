// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "Domain/ElementId.hpp"

namespace LinearSolver::multigrid {

template <size_t Dim>
std::pair<std::vector<std::array<size_t, Dim>>,
          std::vector<std::array<size_t, Dim>>>
coarsen(const std::vector<std::array<size_t, Dim>>& ref_levs_in_all_blocks,
        const std::vector<std::array<size_t, Dim>>& extents_in_all_blocks,
        size_t min_extent) noexcept;

template <size_t Dim>
ElementId<Dim> parent_id(const ElementId<Dim>& element_id) noexcept;

template <size_t Dim>
std::vector<ElementId<Dim>> child_ids(
    const ElementId<Dim>& element_id,
    const std::array<size_t, Dim>& base_ref_levs,
    const std::array<size_t, Dim>& ref_levs) noexcept;

}  // namespace LinearSolver::multigrid
