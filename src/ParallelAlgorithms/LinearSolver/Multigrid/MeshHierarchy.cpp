// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Multigrid/MeshHierarchy.hpp"

#include <array>
#include <boost/range/combine.hpp>
#include <cstddef>
#include <optional>
#include <vector>

#include "Domain/ElementId.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#include "Parallel/Printf.hpp"

namespace LinearSolver::multigrid {

template <size_t Dim>
std::pair<std::vector<std::array<size_t, Dim>>,
          std::vector<std::array<size_t, Dim>>>
coarsen(const std::vector<std::array<size_t, Dim>>& ref_levs,
        const std::vector<std::array<size_t, Dim>>& extents,
        const size_t min_extent) noexcept {
  std::vector<std::array<size_t, Dim>> coarsened_ref_levs = ref_levs;
  std::vector<std::array<size_t, Dim>> coarsened_extents = extents;
  std::transform(ref_levs.begin(), ref_levs.end(), coarsened_ref_levs.begin(),
                 [](const auto& ref_levs_block) noexcept {
                   std::array<size_t, Dim> coarsened_ref_levs_block{};
                   std::transform(ref_levs_block.begin(), ref_levs_block.end(),
                                  coarsened_ref_levs_block.begin(),
                                  [](const size_t ref_lev_block) noexcept {
                                    return ref_lev_block > 0 ? ref_lev_block - 1
                                                             : 0;
                                  });
                   return coarsened_ref_levs_block;
                 });
  for (decltype(auto) ref_levs_and_extents : boost::combine(
           ref_levs, extents, coarsened_ref_levs, coarsened_extents)) {
    const auto& ref_levs_block = boost::get<0>(ref_levs_and_extents);
    const auto& extents_block = boost::get<1>(ref_levs_and_extents);
    const auto& coarsened_ref_levs_block = boost::get<2>(ref_levs_and_extents);
    auto& coarsened_extents_block = boost::get<3>(ref_levs_and_extents);
    if (ref_levs_block == coarsened_ref_levs_block) {
      std::transform(extents_block.begin(), extents_block.end(),
                     coarsened_extents_block.begin(),
                     [&](const size_t extent_block) noexcept {
                       return extent_block > min_extent ? extent_block - 1
                                                        : extent_block;
                     });
    }
  }
  return {std::move(coarsened_ref_levs), std::move(coarsened_extents)};
}

template <size_t Dim>
ElementId<Dim> parent_id(const ElementId<Dim>& element_id) noexcept {
  const auto& segment_ids = element_id.segment_ids();
  std::array<SegmentId, Dim> parent_segment_ids{};
  std::transform(
      segment_ids.begin(), segment_ids.end(), parent_segment_ids.begin(),
      [](const SegmentId& segment_id) noexcept {
        return segment_id.refinement_level() > 0 ? segment_id.id_of_parent()
                                                 : segment_id;
      });
  return ElementId<Dim>{element_id.block_id(), std::move(parent_segment_ids),
                        element_id.grid_index() + 1};
}

template <>
std::vector<ElementId<1>> child_ids<1>(
    const ElementId<1>& element_id, const std::array<size_t, 1>& base_ref_levs,
    const std::array<size_t, 1>& ref_levs) noexcept {
  if (element_id.grid_index() == 0) {
    return {};
  }
  const auto& segment_ids = element_id.segment_ids();
  std::vector<ElementId<1>> child_ids{};
  if (ref_levs[0] < base_ref_levs[0]) {
    child_ids.emplace_back(
        element_id.block_id(),
        std::array<SegmentId, 1>{segment_ids[0].id_of_child(Side::Lower)},
        element_id.grid_index() - 1);
    child_ids.emplace_back(
        element_id.block_id(),
        std::array<SegmentId, 1>{segment_ids[0].id_of_child(Side::Upper)},
        element_id.grid_index() - 1);
  } else {
    child_ids.emplace_back(element_id.block_id(), segment_ids,
                           element_id.grid_index() - 1);
  }
  return child_ids;
}

template <>
std::vector<ElementId<2>> child_ids<2>(
    const ElementId<2>& element_id, const std::array<size_t, 2>& base_ref_levs,
    const std::array<size_t, 2>& ref_levs) noexcept {
  if (element_id.grid_index() == 0) {
    return {};
  }
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
      child_ids.emplace_back(
          element_id.block_id(),
          std::array<SegmentId, 2>{{child_segment_id_x, child_segment_id_y}},
          element_id.grid_index() - 1);
    }
  }
  return child_ids;
}

template <>
std::vector<ElementId<3>> child_ids<3>(
    const ElementId<3>& element_id, const std::array<size_t, 3>& base_ref_levs,
    const std::array<size_t, 3>& ref_levs) noexcept {
  if (element_id.grid_index() == 0) {
    return {};
  }
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
        child_ids.emplace_back(
            element_id.block_id(),
            std::array<SegmentId, 3>{
                {child_segment_id_x, child_segment_id_y, child_segment_id_z}},
            element_id.grid_index() - 1);
      }
    }
  }
  return child_ids;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                     \
  template std::pair<std::vector<std::array<size_t, DIM(data)>>, \
                     std::vector<std::array<size_t, DIM(data)>>> \
  coarsen(const std::vector<std::array<size_t, DIM(data)>>&,     \
          const std::vector<std::array<size_t, DIM(data)>>&,     \
          const size_t) noexcept;                                \
  template ElementId<DIM(data)> parent_id(                       \
      const ElementId<DIM(data)>& element_id) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace LinearSolver::multigrid
