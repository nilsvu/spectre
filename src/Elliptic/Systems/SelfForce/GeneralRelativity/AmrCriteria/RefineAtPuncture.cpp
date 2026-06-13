// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/AmrCriteria/RefineAtPuncture.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeArray.hpp"

namespace GrSelfForce::AmrCriteria {

std::array<amr::Flag, 2> RefineAtPuncture::impl(
    const elliptic::analytic_data::Background& background,
    const Domain<2>& domain, const ElementId<2>& element_id) {
  const auto* co_ptr =
      dynamic_cast<const GrSelfForce::AnalyticData::CircularOrbit*>(
          &background);
  const auto* nd_ptr =
      co_ptr != nullptr
          ? nullptr
          : dynamic_cast<const GrSelfForce::AnalyticData::NumericData*>(
                &background);
  if (co_ptr == nullptr and nd_ptr == nullptr) {
    ERROR("Background must be CircularOrbit or NumericData");
  }
  const auto puncture_position =
      co_ptr != nullptr ? co_ptr->puncture_position()
                        : nd_ptr->puncture_position();
  // Split (h-refine) the element if it contains the puncture
  const auto& block = domain.blocks()[element_id.block_id()];
  // Check if the puncture is in the block
  auto block_logical_coords =
      block_logical_coordinates_single_point(puncture_position, block);
  if (not block_logical_coords.has_value()) {
    return make_array<2>(amr::Flag::DoNothing);
  }
  for (size_t d = 0; d < 2; ++d) {
    if (abs(block_logical_coords->get(d)) < 1e-10) {
      block_logical_coords->get(d) = 0.;
    }
  }
  if (not element_logical_coordinates(*block_logical_coords, element_id)) {
    return make_array<2>(amr::Flag::DoNothing);
  }
  return make_array<2>(amr::Flag::Split);
}

PUP::able::PUP_ID RefineAtPuncture::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::AmrCriteria
