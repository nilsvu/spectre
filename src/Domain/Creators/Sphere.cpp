// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Sphere.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
Sphere::Sphere(
    const double inner_radius, const double outer_radius,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    const bool use_equiangular_map, std::vector<double> radial_partitioning,
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      use_equiangular_map_(use_equiangular_map),
      radial_partitioning_(std::move(radial_partitioning)),
      radial_distribution_(std::move(radial_distribution)),
      boundary_condition_(std::move(boundary_condition)) {
  if (inner_radius_ > outer_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) + " and outer radius is " +
                    std::to_string(outer_radius_) + ".");
  }
  if (not std::is_sorted(radial_partitioning_.begin(),
                         radial_partitioning_.end())) {
    PARSE_ERROR(context,
                "Specify radial partitioning in ascending order. Specified "
                "radial partitioning is: " +
                    get_output(radial_partitioning_));
  }
  if (not radial_partitioning_.empty()) {
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(
          context,
          "First radial partition must be larger than inner radius, but is: " +
              std::to_string(inner_radius_));
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(
          context,
          "Last radial partition must be smaller than outer radius, but is: " +
              std::to_string(outer_radius_));
    }
  }

  const size_t num_shells = 1 + radial_partitioning_.size();
  if (radial_distribution_.size() != num_shells) {
    PARSE_ERROR(context,
                "Specify a 'RadialDistribution' for every spherical shell. You "
                "specified "
                    << radial_distribution_.size()
                    << " items, but the domain has " << num_shells
                    << " shells.");
  }
  if (radial_distribution_.front() !=
      domain::CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(context,
                "The 'RadialDistribution' must be 'Linear' for the innermost "
                "shell because it changes in sphericity. Add entries to "
                "'RadialPartitioning' to add outer shells for which you can "
                "select different radial distributions.");
  }

  // Create block names and groups
  static std::array<std::string, 6> wedge_directions{
      "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX", "LowerX"};
  for (size_t shell = 0; shell < num_shells; ++shell) {
    std::string shell_prefix =
        "Shell" + (num_shells > 1 ? std::to_string(shell) : "");
    for (size_t direction = 0; direction < 6; ++direction) {
      const std::string wedge_name =
          shell_prefix + gsl::at(wedge_directions, direction);
      block_names_.push_back(wedge_name);
      block_groups_[shell_prefix].insert(wedge_name);
    }
  }
  block_names_.push_back("InnerCube");

  // Expand initial refinement and number of grid points over all blocks
  const ExpandOverBlocks<size_t, 3> expand_over_blocks{block_names_,
                                                       block_groups_};
  try {
    initial_refinement_ = std::visit(expand_over_blocks, initial_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }
  try {
    initial_number_of_grid_points_ =
        std::visit(expand_over_blocks, initial_number_of_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }

  // The central cube has no notion of a "radial" direction, so we set
  // refinement and number of grid points of the central cube z direction to its
  // y value, which corresponds to the azimuthal direction of the wedges. This
  // keeps the boundaries conforming when the radial direction is chosen
  // differently to the angular directions.
  auto& central_cube_refinement = initial_refinement_.back();
  auto& central_cube_grid_points = initial_number_of_grid_points_.back();
  central_cube_refinement[2] = central_cube_refinement[1];
  central_cube_grid_points[2] = central_cube_grid_points[1];

  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_)) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with a Sphere");
  }
}

Domain<3> Sphere::create_domain() const {
  const size_t num_shells = 1 + radial_partitioning_.size();
  std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(num_shells, true);

  auto coord_maps = domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                            Frame::Inertial, 3>(
      sph_wedge_coordinate_maps(inner_radius_, outer_radius_, 0.0, 1.0,
                                use_equiangular_map_, false,
                                radial_partitioning_, radial_distribution_));
  if (use_equiangular_map_) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Equiangular3D{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0))}));
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_ != nullptr) {
    boundary_conditions_all_blocks.resize(coord_maps.size());
    // The last block is the inner cube. The six preceding blocks are the wedges
    // of the outer shell where we want to set the boundary condition.
    for (size_t block_id = coord_maps.size() - 7;
         block_id < coord_maps.size() - 1; ++block_id) {
      boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
          boundary_condition_->get_clone();
    }
  }

  return Domain<3>(std::move(coord_maps), corners, {},
                   std::move(boundary_conditions_all_blocks));
}
}  // namespace domain::creators
