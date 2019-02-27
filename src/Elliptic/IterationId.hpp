// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {

/*!
 * \brief The factor to divide the `elliptic::iteration_id` by to
 * recover the step number of the `ComponentTag`
 *
 * \see elliptic::iteration_id
 */
template <typename ComponentTag>
constexpr size_t iteration_id_value_factor = 0.;
template <>
constexpr size_t iteration_id_value_factor<LinearSolver::Tags::IterationId> = 1;
template <>
constexpr size_t iteration_id_value_factor<NonlinearSolver::Tags::IterationId> =
    1e6;

/*!
 * \brief Encodes all components to identify a step in an elliptic solve by a
 * single number.
 *
 * \details Since we must encode the various iteration ids used in an elliptic
 * solve in a single number to identify observations, we embed each component
 * into a specific range within a `double`...
 *
 * For each component we allocate a fixed range of integers specified ...
 *
 * For now assume the max number of steps for each component, then we can
 * reconstruct from the double. Eventually, use one size_t to identify the
 * total step, and construct it from each component's number of sub-steps
 * (which must then be kept track of).
 */
template <typename... ComponentTags>
double iteration_id(
    const db::item_type<ComponentTags>&... component_ids) noexcept {
  size_t combined_id = 0;
  const auto helper = [&combined_id](const auto component_tag_v,
                                     const auto component_id) noexcept {
    using component_tag = std::decay_t<decltype(component_tag_v)>;
    combined_id += iteration_id_value_factor<component_tag> * component_id;
  };
  EXPAND_PACK_LEFT_TO_RIGHT(helper(ComponentTags{}, component_ids));
  return combined_id;
}

namespace Tags {
/*!
 * \brief Encodes all components to identify a step in an elliptic solve by a
 * single number.
 *
 * \see elliptic::iteration_id
 */
template <typename... ComponentTags>
struct IterationIdCompute : db::ComputeTag, IterationId {
  using argument_tags = tmpl::list<ComponentTags...>;
  static constexpr auto function = iteration_id<ComponentTags...>;
};
}  // namespace Tags

}  // namespace elliptic
