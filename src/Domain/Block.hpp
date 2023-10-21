// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Block.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_set>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Structure/BlockGeometry.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"

/// \cond
namespace Frame {
struct BlockLogical;
struct Inertial;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// A Block<VolumeDim> is a region of a VolumeDim-dimensional computational
/// domain that defines the root node of a tree which is used to construct the
/// Elements that cover a region of the computational domain.
///
/// Each codimension 1 boundary of a Block<VolumeDim> is either an external
/// boundary or an internal boundary to one or more neighboring blocks. The only
/// currently supported case where blocks can have more than one neighbor are
/// interfaces between a spherical shell and wedges. In all other cases the
/// internal boundaries between neighboring blocks must be identical.
///
/// A Block has logical coordinates that go from -1 to +1 in each
/// dimension.  The global coordinates are obtained from the logical
/// coordinates from the Coordinatemap:  CoordinateMap::operator() takes
/// Points in the BlockLogical Frame (i.e., block logical coordinates) and
/// returns Points in the Inertial Frame (i.e., the global coordinate frame in
/// which the problem to be solved is set up).
template <size_t VolumeDim>
class Block {
 public:
  /// Block with one neighbor per direction. Currently always a deformed cube.
  ///
  /// \param stationary_map the CoordinateMap.
  /// \param id a unique ID.
  /// \param neighbors info about the Blocks that share a codimension 1
  /// boundary with this Block.
  /// \param name Human-readable name for the block
  Block(std::unique_ptr<domain::CoordinateMapBase<
            Frame::BlockLogical, Frame::Inertial, VolumeDim>>&& stationary_map,
        size_t id, DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbors,
        std::string name = "");

  /// Block with multiple neighbors per direction. Currently only supports
  /// spherical shells.
  Block(std::unique_ptr<domain::CoordinateMapBase<
            Frame::BlockLogical, Frame::Inertial, VolumeDim>>&& stationary_map,
        size_t id,
        DirectionMap<VolumeDim, std::unordered_set<BlockNeighbor<VolumeDim>>>
            neighbors,
        std::string name = "");

  Block() = default;
  ~Block() = default;
  Block(const Block&) = delete;
  Block(Block&&) = default;
  Block& operator=(const Block&) = delete;
  Block& operator=(Block&&) = default;

  domain::BlockGeometry geometry() const { return geometry_; }

  /// \brief The map used when the coordinate map is time-independent.
  ///
  /// \see is_time_dependent()
  const domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial,
                                  VolumeDim>&
  stationary_map() const;

  /// \brief The map going from the block logical frame to the last time
  /// independent frame. Only used when the coordinate map is time-dependent.
  ///
  /// \see is_time_dependent() moving_mesh_grid_to_inertial_map()
  const domain::CoordinateMapBase<Frame::BlockLogical, Frame::Grid, VolumeDim>&
  moving_mesh_logical_to_grid_map() const;

  /// \brief The map going from the last time independent frame to the frame in
  /// which the equations are solved. Only used when the coordinate map is
  /// time-dependent.
  ///
  /// \see is_time_dependent() moving_mesh_logical_to_grid_map()
  const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
  moving_mesh_grid_to_inertial_map() const;

  /// \brief The map going from the last time independent frame to the
  /// distorted frame. Only used when the coordinate map is
  /// time-dependent. See \ref domain_concepts to see how the distorted
  /// frame is defined.
  ///
  /// \see is_time_dependent() moving_mesh_distorted_to_grid_map()
  const domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, VolumeDim>&
  moving_mesh_grid_to_distorted_map() const;

  /// \brief The map going from the distorted frame to the frame in
  /// which the equations are solved. Only used when the coordinate map is
  /// time-dependent. See \ref domain_concepts to see how the distorted
  /// frame is defined.
  ///
  /// \see is_time_dependent() moving_mesh_grid_to_distorted_map()
  const domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, VolumeDim>&
  moving_mesh_distorted_to_inertial_map() const;

  /// \brief Returns `true` if the block has time-dependent maps.
  bool is_time_dependent() const { return stationary_map_ == nullptr; }

  /// \brief Returns `true` if the block has a distorted frame.
  ///
  /// If a block has a distorted frame, then
  ///   - moving_mesh_grid_to_distorted_map() is non-null
  ///   - moving_mesh_distorted_to_inertial_map() is non-null
  ///   - moving_mesh_grid_to_inertial_map() is non-null
  /// Note in particular the last point above:  If the block is time-dependent,
  /// then the block must have a grid_to_inertial map independent of whether
  /// it has a distorted frame.  This allows us to write more efficient maps.
  /// In particular, we often care only about the grid_to_inertial map, so we
  /// can code that map directly instead of composing
  /// grid_to_distorted + distorted_to_inertial maps at runtime.
  ///
  /// If a block does not have a distorted frame, then
  ///   - moving_mesh_grid_to_distorted_map() is null
  ///   - moving_mesh_distorted_to_inertial_map() is null
  ///   - moving_mesh_grid_to_inertial_map() is non-null
  ///   - If we ever find ourselves needing ::Frame::Distorted coordinates
  ///     in that block, we can assume that ::Frame::Distorted and ::Frame::Grid
  ///     are the same.  Usually this case will not occur.
  bool has_distorted_frame() const {
    return moving_mesh_grid_to_distorted_map_ != nullptr and
           moving_mesh_distorted_to_inertial_map_ != nullptr;
  }

  /// \brief Given a Block that has a time-independent map, injects the
  /// time-dependent map into the Block.
  void inject_time_dependent_map(
      std::unique_ptr<
          domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
          moving_mesh_grid_to_inertial_map,
      std::unique_ptr<
          domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, VolumeDim>>
          moving_mesh_grid_to_distorted_map = nullptr,
      std::unique_ptr<domain::CoordinateMapBase<Frame::Distorted,
                                                Frame::Inertial, VolumeDim>>
          moving_mesh_distorted_to_inertial_map = nullptr);

  /// A unique identifier for the Block that is in the range
  /// [0, number_of_blocks -1] where number_of_blocks is the number
  /// of Blocks that cover the computational domain.
  size_t id() const { return id_; }

  /// Information about the neighboring Blocks.
  const DirectionMap<VolumeDim, std::unordered_set<BlockNeighbor<VolumeDim>>>&
  neighbors() const {
    return neighbors_;
  }

  /// The directions of the faces of the Block that are external boundaries.
  const std::unordered_set<Direction<VolumeDim>>& external_boundaries() const {
    return external_boundaries_;
  }

  const std::string& name() const { return name_; }

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalVolumeDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Block<LocalVolumeDim>& lhs,
                         const Block<LocalVolumeDim>& rhs);

  std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,
                                            Frame::Inertial, VolumeDim>>
      stationary_map_{nullptr};
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Grid, VolumeDim>>
      moving_mesh_logical_to_grid_map_{nullptr};
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
      moving_mesh_grid_to_inertial_map_{nullptr};
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, VolumeDim>>
      moving_mesh_grid_to_distorted_map_{nullptr};
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, VolumeDim>>
      moving_mesh_distorted_to_inertial_map_{nullptr};

  size_t id_{0};
  domain::BlockGeometry geometry_{domain::BlockGeometry::Cube};
  DirectionMap<VolumeDim, std::unordered_set<BlockNeighbor<VolumeDim>>>
      neighbors_;
  std::unordered_set<Direction<VolumeDim>> external_boundaries_;
  std::string name_;
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Block<VolumeDim>& block);

template <size_t VolumeDim>
bool operator!=(const Block<VolumeDim>& lhs, const Block<VolumeDim>& rhs);
