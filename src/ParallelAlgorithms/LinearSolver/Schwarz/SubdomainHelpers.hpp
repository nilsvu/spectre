// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Domain/OrientationMapHelpers.hpp"

namespace LinearSolver {
namespace schwarz_detail {

template <typename TagsList>
Variables<TagsList> data_on_overlap(const Variables<TagsList>& vars) noexcept {
  return vars;
}

template <typename TagsList>
Variables<TagsList> orient_data_on_overlap(
    const Variables<TagsList>& vars) noexcept {
  return vars;
}

}  // namespace schwarz_detail
}  // namespace LinearSolver
