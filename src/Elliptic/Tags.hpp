// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"

namespace elliptic {

/// The \ref DataBoxGroup tags for elliptic solves
namespace Tags {

struct FluxComputerBase : db::BaseTag {};

template <typename FluxType>
struct FluxComputer : db::SimpleTag, FluxComputerBase {
  using type = FluxType;
  static std::string name() noexcept { return "FluxComputer"; }
};

/*!
 * \brief Identifies a step in an elliptic solve.
 *
 * \see elliptic::iteration_id
 */
struct IterationId : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "EllipticIterationId"; }
};

}  // namespace Tags

}  // namespace elliptic
