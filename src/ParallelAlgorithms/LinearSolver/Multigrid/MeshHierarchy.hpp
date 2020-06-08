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
bool can_coarsen(const std::vector<std::array<size_t, Dim>>&
                     ref_levs_in_all_blocks) noexcept;

template <size_t Dim>
std::vector<std::array<size_t, Dim>> coarsen(
    std::vector<std::array<size_t, Dim>> ref_levs_in_all_blocks) noexcept;

template <size_t Dim>
std::optional<ElementId<Dim>> parent_id(
    const ElementId<Dim>& element_id) noexcept;

template <size_t Dim>
std::vector<ElementId<Dim>> child_ids(
    const ElementId<Dim>& element_id,
    const std::array<size_t, Dim>& base_ref_levs,
    const std::array<size_t, Dim>& ref_levs) noexcept;

}  // namespace LinearSolver::multigrid
