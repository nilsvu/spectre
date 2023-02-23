// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Interval.hpp"

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
Interval::Interval(
    std::array<double, 1> lower_x, std::array<double, 1> upper_x,
    std::array<size_t, 1> initial_refinement_level_x,
    std::array<size_t, 1> initial_number_of_grid_points_in_x,
    std::array<bool, 1> is_periodic_in_x,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
        time_dependence)
    : lower_x_(lower_x),
      upper_x_(upper_x),
      is_periodic_in_x_(is_periodic_in_x),
      initial_refinement_level_x_(initial_refinement_level_x),
      initial_number_of_grid_points_in_x_(initial_number_of_grid_points_in_x),
      lower_boundary_condition_(nullptr),
      upper_boundary_condition_(nullptr),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<1>>();
  }
}

Interval::Interval(
    std::array<double, 1> lower_x, std::array<double, 1> upper_x,
    std::array<size_t, 1> initial_refinement_level_x,
    std::array<size_t, 1> initial_number_of_grid_points_in_x,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        lower_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        upper_boundary_condition,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
        time_dependence,
    const Options::Context& context)
    : lower_x_(lower_x),
      upper_x_(upper_x),
      is_periodic_in_x_{{false}},
      initial_refinement_level_x_(initial_refinement_level_x),
      initial_number_of_grid_points_in_x_(initial_number_of_grid_points_in_x),
      lower_boundary_condition_(std::move(lower_boundary_condition)),
      upper_boundary_condition_(std::move(upper_boundary_condition)),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<1>>();
  }
  if ((lower_boundary_condition_ == nullptr) !=
      (upper_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Both upper and lower boundary conditions must be specified, "
                "or neither.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(lower_boundary_condition_) or
      is_none(upper_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(lower_boundary_condition_) !=
      is_periodic(upper_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Both the upper and lower boundary condition must be set to periodic "
        "if imposing periodic boundary conditions.");
  }
  if (is_periodic(lower_boundary_condition_)) {
    is_periodic_in_x_[0] = true;
  }
}

Domain<1> Interval::create_domain() const {
  Domain<1> domain{
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          CoordinateMaps::Affine{-1., 1., lower_x_[0], upper_x_[0]}),
      std::vector<std::array<size_t, 2>>{{{1, 2}}},
      is_periodic_in_x_[0] ? std::vector<PairOfFaces>{{{1}, {2}}}
                           : std::vector<PairOfFaces>{}};
  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps_grid_to_inertial(1)[0]),
        std::move(time_dependence_->block_maps_grid_to_distorted(1)[0]),
        std::move(time_dependence_->block_maps_distorted_to_inertial(1)[0]));
  }
  return domain;
}

std::vector<DirectionMap<
    1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
Interval::external_boundary_conditions() const {
  if (upper_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{1};
  if (is_periodic_in_x_[0]) {
    return boundary_conditions;
  }
  boundary_conditions[0][Direction<1>::lower_xi()] =
      lower_boundary_condition_->get_clone();
  boundary_conditions[0][Direction<1>::upper_xi()] =
      upper_boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::array<size_t, 1>> Interval::initial_extents() const {
  return {{{initial_number_of_grid_points_in_x_}}};
}

std::vector<std::array<size_t, 1>> Interval::initial_refinement_levels() const {
  return {{{initial_refinement_level_x_}}};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Interval::functions_of_time(const std::unordered_map<std::string, double>&
                                initial_expiration_times) const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators
