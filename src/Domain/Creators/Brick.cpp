// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Brick.hpp"

#include <array>
#include <memory>
#include <vector>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep

namespace Frame {
struct Logical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {
Brick::Brick(
    typename LowerBound::type lower_xyz, typename UpperBound::type upper_xyz,
    typename InitialRefinement::type initial_refinement_level_xyz,
    typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
    typename IsPeriodicIn::type is_periodic_in_xyz,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence) noexcept
    // clang-tidy: trivially copyable
    : lower_xyz_(std::move(lower_xyz)),                      // NOLINT
      upper_xyz_(std::move(upper_xyz)),                      // NOLINT
      is_periodic_in_xyz_(std::move(is_periodic_in_xyz)),    // NOLINT
      initial_refinement_level_xyz_(                         // NOLINT
          std::move(initial_refinement_level_xyz)),          // NOLINT
      initial_number_of_grid_points_in_xyz_(                 // NOLINT
          std::move(initial_number_of_grid_points_in_xyz)),  // NOLINT
      time_dependence_(std::move(time_dependence)),
      boundary_condition_lower_z_(nullptr) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  block_names_.push_back("Brick");
}

Brick::Brick(
    typename LowerBound::type lower_xyz, typename UpperBound::type upper_xyz,
    typename InitialRefinement::type initial_refinement_level_xyz,
    typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_lower_z,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_remaining,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    const Options::Context& context)
    : Brick(lower_xyz, upper_xyz, initial_refinement_level_xyz,
            initial_number_of_grid_points_in_xyz, {{false, false, false}},
            std::move(time_dependence)) {
  boundary_condition_lower_z_ = std::move(boundary_condition_lower_z);
  boundary_condition_remaining_ = std::move(boundary_condition_remaining);
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_lower_z_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_lower_z_)) {
    is_periodic_in_xyz_[0] = true;
    is_periodic_in_xyz_[1] = true;
    is_periodic_in_xyz_[2] = true;
    boundary_condition_lower_z_ = nullptr;
  }
}

Domain<3> Brick::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xyz_[0]) {
    identifications.push_back({{0, 4, 2, 6}, {1, 5, 3, 7}});
  }
  if (is_periodic_in_xyz_[1]) {
    identifications.push_back({{0, 1, 4, 5}, {2, 3, 6, 7}});
  }
  if (is_periodic_in_xyz_[2]) {
    identifications.push_back({{0, 1, 2, 3}, {4, 5, 6, 7}});
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_lower_z_ != nullptr) {
    DirectionMap<3,
                 std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        boundary_conditions{};
    for (const auto& direction : Direction<3>::all_directions()) {
      if (direction == Direction<3>::lower_zeta()) {
        boundary_conditions[direction] =
            boundary_condition_lower_z_->get_clone();
      } else {
        boundary_conditions[direction] =
            boundary_condition_remaining_->get_clone();
      }
    }
    boundary_conditions_all_blocks.push_back(std::move(boundary_conditions));
  }

  Domain<3> domain{
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine3D{Affine{-1., 1., lower_xyz_[0], upper_xyz_[0]},
                   Affine{-1., 1., lower_xyz_[1], upper_xyz_[1]},
                   Affine{-1., 1., lower_xyz_[2], upper_xyz_[2]}}),
      std::vector<std::array<size_t, 8>>{{{0, 1, 2, 3, 4, 5, 6, 7}}},
      identifications, std::move(boundary_conditions_all_blocks)};

  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps(1)[0]));
  }
  return domain;
}

std::vector<std::array<size_t, 3>> Brick::initial_extents() const noexcept {
  return {initial_number_of_grid_points_in_xyz_};
}

std::vector<std::array<size_t, 3>> Brick::initial_refinement_levels() const
    noexcept {
  return {initial_refinement_level_xyz_};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Brick::functions_of_time() const noexcept {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}

std::vector<std::string> Brick::block_names() const noexcept {
  return block_names_;
}

std::unordered_map<std::string, std::unordered_set<std::string>>
Brick::block_groups() const noexcept {
  return block_groups_;
}
}  // namespace domain::creators
