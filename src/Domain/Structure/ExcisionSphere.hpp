// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ExcisionSphere.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iosfwd>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// The excision sphere information of a computational domain.
/// The excision sphere is assumed to be a coordinate sphere in the
/// grid frame.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class ExcisionSphere {
 private:
  using BlockId = std::pair<size_t, Direction<VolumeDim>>;

 public:
  /// Constructor
  ///
  /// \param radius the radius of the excision sphere in the
  /// computational domain.
  /// \param center the coordinate center of the excision sphere
  /// in the computational domain.
  ExcisionSphere(
      double radius, std::array<double, VolumeDim> center,
      std::unordered_set<BlockId, boost::hash<BlockId>> block_neighbors);

  /// Default constructor needed for Charm++ serialization.
  ExcisionSphere() = default;
  ~ExcisionSphere() = default;
  ExcisionSphere(const ExcisionSphere<VolumeDim>& /*rhs*/) = default;
  ExcisionSphere(ExcisionSphere<VolumeDim>&& /*rhs*/) = default;
  ExcisionSphere<VolumeDim>& operator=(
      const ExcisionSphere<VolumeDim>& /*rhs*/) = default;
  ExcisionSphere<VolumeDim>& operator=(ExcisionSphere<VolumeDim>&& /*rhs*/) =
      default;

  /// The radius of the ExcisionSphere.
  double radius() const { return radius_; }

  /// The coodinate center of the ExcisionSphere.
  const std::array<double, VolumeDim>& center() const { return center_; }

  const auto& block_neighbors() const { return block_neighbors_; }

  std::optional<Direction<VolumeDim>> radial_direction(
      const size_t block_id) const;
  std::optional<Direction<VolumeDim>> radial_direction(
      const ElementId<VolumeDim>& element_id) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, VolumeDim> center_{
      make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN())};
  std::unordered_set<BlockId, boost::hash<BlockId>> block_neighbors_{};
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const ExcisionSphere<VolumeDim>& excision_sphere);

template <size_t VolumeDim>
bool operator==(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs);
