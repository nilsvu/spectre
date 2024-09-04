// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DomainCreator.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/DirectionMap.hpp"

/// \cond
namespace Frame {
struct Grid;
}  // namespace Frame
template <size_t>
class Domain;
namespace domain::BoundaryConditions {
class BoundaryCondition;
}
/// \endcond

namespace domain {
/// \ingroup ComputationalDomainGroup
/// \brief Defines classes that create Domains.
namespace creators {}
}  // namespace domain

/// \ingroup ComputationalDomainGroup
/// \brief Base class for creating Domains from an option string.
template <size_t VolumeDim>
class DomainCreator {
 public:
  static constexpr size_t volume_dim = VolumeDim;

  DomainCreator() = default;
  DomainCreator(const DomainCreator<VolumeDim>&) = delete;
  DomainCreator(DomainCreator<VolumeDim>&&) = default;
  DomainCreator<VolumeDim>& operator=(const DomainCreator<VolumeDim>&) = delete;
  DomainCreator<VolumeDim>& operator=(DomainCreator<VolumeDim>&&) = default;
  virtual ~DomainCreator() = default;

  virtual Domain<VolumeDim> create_domain() const = 0;

  /// A set of named coordinates in the grid frame, like the center of the
  /// domain or the positions of specific objects in a domain
  virtual std::unordered_map<std::string,
                             tnsr::I<double, VolumeDim, Frame::Grid>>
  grid_anchors() const {
    return {};
  }

  /// The set of external boundary condition for every block in the domain
  virtual std::vector<DirectionMap<
      VolumeDim,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const = 0;

  /// A human-readable name for every block, or empty if the domain creator
  /// doesn't support block names (yet).
  virtual std::vector<std::string> block_names() const {
    const auto domain = create_domain();
    const auto& blocks = domain.blocks();
    std::vector<std::string> names;
    names.reserve(blocks.size());
    for (size_t block_id = 0; block_id < blocks.size(); ++block_id) {
      names.push_back("Block" + std::to_string(block_id));
    }
    return names;
  }

  /// Labels to refer to groups of blocks. The groups can overlap, and they
  /// don't have to cover all blocks in the domain. The groups can be used to
  /// refer to multiple blocks at once when specifying input-file options.
  virtual std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const {
    return {};
  }

  /// Obtain the initial grid extents of the Element%s in each block.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_extents()
      const = 0;

  /// Obtain the initial refinement levels of the blocks.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_refinement_levels()
      const = 0;

  /// Retrieve the functions of time used for moving meshes.
  // LCOV_EXCL_START
  virtual auto functions_of_time(const std::unordered_map<std::string, double>&
                                     initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> {
    (void)(initial_expiration_times);
    return {};
  }
  // LCOV_EXCL_STOP
};
