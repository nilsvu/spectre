// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ExcisionSphere.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <pup_stl.h>
#include <unordered_set>
#include <utility>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
ExcisionSphere<VolumeDim>::ExcisionSphere(
    const double radius, const std::array<double, VolumeDim> center,
    std::unordered_set<BlockId, boost::hash<BlockId>> block_neighbors)
    : radius_(radius),
      center_(center),
      block_neighbors_(std::move(block_neighbors)) {
  ASSERT(radius_ > 0.0,
         "The ExcisionSphere must have a radius greater than zero.");
}

template <size_t VolumeDim>
std::optional<Direction<VolumeDim>> ExcisionSphere<VolumeDim>::radial_direction(
    const size_t block_id) const {
  for (const auto& [local_block_id, local_radial_direction] :
       block_neighbors_) {
    if (block_id == local_block_id) {
      return local_radial_direction;
    }
  }
  return std::nullopt;
}

template <size_t VolumeDim>
std::optional<Direction<VolumeDim>> ExcisionSphere<VolumeDim>::radial_direction(
    const ElementId<VolumeDim>& element_id) const {
  const auto& found_direction = radial_direction(element_id.block_id());
  if (not found_direction.has_value()) {
    return std::nullopt;
  }
  const Direction<VolumeDim>& direction = found_direction.value();
  const auto& radial_segment_id = element_id.segment_id(direction.dimension());
  ASSERT(direction.side() == Side::Lower,
         "Implemented only for Side::Lower radial direction at the moment.");
  if (radial_segment_id.index() == 0) {
    return direction;
  } else {
    return std::nullopt;
  }
}

template <size_t VolumeDim>
void ExcisionSphere<VolumeDim>::pup(PUP::er& p) {
  p | radius_;
  p | center_;
  p | block_neighbors_;
}

template <size_t VolumeDim>
bool operator==(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) {
  return lhs.radius() == rhs.radius() and lhs.center() == rhs.center() and
         lhs.block_neighbors() == rhs.block_neighbors();
}

template <size_t VolumeDim>
bool operator!=(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const ExcisionSphere<VolumeDim>& sphere) {
  os << "ExcisionSphere:\n";
  os << "  Radius: " << sphere.radius() << "\n";
  os << "  Center: " << sphere.center() << "\n";
  os << "  Block neighbors: " << sphere.block_neighbors() << "\n";
  return os;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                    \
  template class ExcisionSphere<GET_DIM(data)>;                   \
  template bool operator==(const ExcisionSphere<GET_DIM(data)>&,  \
                           const ExcisionSphere<GET_DIM(data)>&); \
  template bool operator!=(const ExcisionSphere<GET_DIM(data)>&,  \
                           const ExcisionSphere<GET_DIM(data)>&); \
  template std::ostream& operator<<(std::ostream&,                \
                                    const ExcisionSphere<GET_DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
