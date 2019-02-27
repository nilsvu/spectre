// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"

namespace Elliptic {

/*!
 * \brief The \ref DataBoxGroup tags for elliptic solves
 */
namespace Tags {

/*!
 * \brief Holds an `Elliptic::IterationId` that identifies a step in the
 * elliptic solver algorithm
 */
template <typename... ComponentTags>
struct IterationId : db::SimpleTag {
  using type = Elliptic::IterationId<ComponentTags...>;
  static std::string name() noexcept { return "EllipticIterationId"; }
  template <typename Tag>
  using step_prefix =
      typename LinearSolver::Tags::IterationId::template step_prefix<Tag>;
};

template <typename... ComponentTags>
struct IterationIdCompute : db::ComputeTag, IterationId<ComponentTags...> {
  using argument_tags = tmpl::list<ComponentTags...>;
  static Elliptic::IterationId<ComponentTags...> function(
      const db::item_type<ComponentTags>&... component_ids) noexcept {
    return {component_ids...};
  }
};

template <typename IdTag>
struct NextCompute : db::ComputeTag, ::Tags::Next<IdTag> {
  using argument_tags = tmpl::list<IdTag>;
  static db::item_type<IdTag> function(
      const db::item_type<IdTag>& id) noexcept {
    return id.next();
  }
};

}  // namespace Tags

namespace OptionTags {
template <typename NumericalFluxType>
struct NumericalFlux {
  static constexpr OptionString help = "The options for the numerical flux";
  using type = NumericalFluxType;
};
template <typename NumericalFluxType>
struct CorrectionNumericalFlux {
  static constexpr OptionString help = "The options for the numerical flux";
  using type = NumericalFluxType;
  static std::string name() noexcept { return "CorrectionNumFlux"; }
};
}  // namespace OptionTags

}  // namespace Elliptic
