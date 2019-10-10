// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"

namespace orbital {
namespace Tags {

struct CenterOfMass : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "CenterOfMass"; }
};

struct AngularVelocity : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "AngularVelocity"; }
};

}  // namespace Tags
}  // namespace orbital
