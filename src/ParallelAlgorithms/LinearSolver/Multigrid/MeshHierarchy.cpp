// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Multigrid/MeshHierarchy.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "Domain/ElementId.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#include "Parallel/Printf.hpp"

namespace LinearSolver::multigrid {

template <size_t Dim>
bool can_coarsen(const std::vector<std::array<size_t, Dim>>&
                     ref_levs_in_all_blocks) noexcept {
  return std::any_of(
      ref_levs_in_all_blocks.begin(), ref_levs_in_all_blocks.end(),
      [](const auto& ref_levs) noexcept {
        return std::any_of(
            ref_levs.begin(), ref_levs.end(),
            [](const size_t ref_lev) noexcept { return ref_lev > 0; });
      });
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>> coarsen(
    std::vector<std::array<size_t, Dim>> ref_levs_in_all_blocks) noexcept {
  std::transform(ref_levs_in_all_blocks.begin(), ref_levs_in_all_blocks.end(),
                 ref_levs_in_all_blocks.begin(),
                 [](const auto& ref_levs) noexcept {
                   std::array<size_t, Dim> coarser_ref_levs{};
                   std::transform(ref_levs.begin(), ref_levs.end(),
                                  coarser_ref_levs.begin(),
                                  [](const size_t ref_lev) noexcept {
                                    return ref_lev > 0 ? ref_lev - 1 : 0;
                                  });
                   return coarser_ref_levs;
                 });
  return ref_levs_in_all_blocks;
}

template <size_t Dim>
std::optional<ElementId<Dim>> parent_id(
    const ElementId<Dim>& element_id) noexcept {
  const auto& segment_ids = element_id.segment_ids();
  std::array<SegmentId, Dim> parent_segment_ids{};
  std::transform(
      segment_ids.begin(), segment_ids.end(), parent_segment_ids.begin(),
      [](const SegmentId& segment_id) noexcept {
        return segment_id.refinement_level() > 0 ? segment_id.id_of_parent()
                                                 : segment_id;
      });
  return parent_segment_ids != element_id.segment_ids()
             ? std::make_optional(ElementId<Dim>{element_id.block_id(),
                                                 std::move(parent_segment_ids)})
             : std::nullopt;
}

template <>
std::vector<ElementId<1>> child_ids<1>(
    const ElementId<1>& element_id, const std::array<size_t, 1>& base_ref_levs,
    const std::array<size_t, 1>& ref_levs) noexcept {
  const auto& segment_ids = element_id.segment_ids();
  std::vector<ElementId<1>> child_ids{};
  if (ref_levs[0] < base_ref_levs[0]) {
    child_ids.emplace_back(
        element_id.block_id(),
        std::array<SegmentId, 1>{segment_ids[0].id_of_child(Side::Lower)});
    child_ids.emplace_back(
        element_id.block_id(),
        std::array<SegmentId, 1>{segment_ids[0].id_of_child(Side::Upper)});
  }
  return child_ids;
}

template <>
std::vector<ElementId<2>> child_ids<2>(
    const ElementId<2>& element_id, const std::array<size_t, 2>& base_ref_levs,
    const std::array<size_t, 2>& ref_levs) noexcept {
  const auto& segment_ids = element_id.segment_ids();
  std::vector<ElementId<2>> child_ids{};
  std::array<std::vector<SegmentId>, 2> child_segment_ids{};
  for (size_t d = 0; d < 2; d++) {
    child_segment_ids[d] =
        ref_levs[d] < base_ref_levs[d]
            ? std::vector<SegmentId>{segment_ids[d].id_of_child(Side::Lower),
                                     segment_ids[d].id_of_child(Side::Upper)}
            : std::vector<SegmentId>{segment_ids[d]};
  }
  for (const auto& child_segment_id_x : child_segment_ids[0]) {
    for (const auto& child_segment_id_y : child_segment_ids[1]) {
      if (child_segment_id_x != segment_ids[0] or
          child_segment_id_y != segment_ids[1]) {
        child_ids.emplace_back(
            element_id.block_id(),
            std::array<SegmentId, 2>{{child_segment_id_x, child_segment_id_y}});
      }
    }
  }
  return child_ids;
}

template <>
std::vector<ElementId<3>> child_ids<3>(
    const ElementId<3>& element_id, const std::array<size_t, 3>& base_ref_levs,
    const std::array<size_t, 3>& ref_levs) noexcept {
  const auto& segment_ids = element_id.segment_ids();
  std::vector<ElementId<3>> child_ids{};
  std::array<std::vector<SegmentId>, 3> child_segment_ids{};
  for (size_t d = 0; d < 3; d++) {
    child_segment_ids[d] =
        ref_levs[d] < base_ref_levs[d]
            ? std::vector<SegmentId>{segment_ids[d].id_of_child(Side::Lower),
                                     segment_ids[d].id_of_child(Side::Upper)}
            : std::vector<SegmentId>{segment_ids[d]};
  }
  for (const auto& child_segment_id_x : child_segment_ids[0]) {
    for (const auto& child_segment_id_y : child_segment_ids[1]) {
      for (const auto& child_segment_id_z : child_segment_ids[2]) {
        if (child_segment_id_x != segment_ids[0] or
            child_segment_id_y != segment_ids[1] or
            child_segment_id_z != segment_ids[2]) {
          child_ids.emplace_back(
              element_id.block_id(),
              std::array<SegmentId, 3>{{child_segment_id_x, child_segment_id_y,
                                        child_segment_id_z}});
        }
      }
    }
  }
  return child_ids;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                                  \
  template bool can_coarsen(                                                  \
      const std::vector<std::array<size_t, DIM(data)>>& element_id) noexcept; \
  template std::vector<std::array<size_t, DIM(data)>> coarsen(                \
      std::vector<std::array<size_t, DIM(data)>>                              \
          ref_levs_in_all_blocks) noexcept;                                   \
  template std::optional<ElementId<DIM(data)>> parent_id(                     \
      const ElementId<DIM(data)>& element_id) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace LinearSolver::multigrid
