// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObject.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {

bool BinaryCompactObject::Object::is_excised() const {
  return inner_boundary_condition.has_value();
}

// Time-independent constructor
BinaryCompactObject::BinaryCompactObject(
    Object object_A, Object object_B, const double radius_enveloping_cube,
    const double outer_radius_domain,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    const bool use_projective_map, const double frustum_sphericity,
    const std::optional<double>& radius_enveloping_sphere,
    const CoordinateMaps::Distribution radial_distribution_outer_shell,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : object_A_(std::move(object_A)),
      object_B_(std::move(object_B)),
      radius_enveloping_cube_(radius_enveloping_cube),
      outer_radius_domain_(outer_radius_domain),
      use_projective_map_(use_projective_map),
      frustum_sphericity_(frustum_sphericity),
      radial_distribution_outer_shell_(radial_distribution_outer_shell),
      outer_boundary_condition_(std::move(outer_boundary_condition)) {
  // Determination of parameters for domain construction:
  translation_ = 0.5 * (object_B_.x_coord + object_A_.x_coord);
  length_inner_cube_ = abs(object_A_.x_coord - object_B_.x_coord);
  length_outer_cube_ = 2.0 * radius_enveloping_cube_ / sqrt(3.0);
  if (use_projective_map_) {
    projective_scale_factor_ = length_inner_cube_ / length_outer_cube_;
  } else {
    projective_scale_factor_ = 1.0;
  }
  need_cube_to_sphere_transition_ =
      frustum_sphericity != 1.0 or radius_enveloping_sphere.has_value();

  // Calculate number of blocks
  // Layers 1, 2, 3, 4, and 5 have 12, 12, 10, 10, and 10 blocks, respectively,
  // for a total of 44, or 54 when the cube-to-sphere transition is needed.
  number_of_blocks_ = 44;
  if (need_cube_to_sphere_transition_) {
    number_of_blocks_ += 10;
  }

  // For each object whose interior is not excised, add 1 block
  if (not object_A_.is_excised()) {
    number_of_blocks_++;
  }
  if (not object_B_.is_excised()) {
    number_of_blocks_++;
  }

  if (object_A_.x_coord >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of ObjectA's center is expected to be negative.");
  }
  if (object_B_.x_coord <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of ObjectB's center is expected to be positive.");
  }
  if (length_outer_cube_ <= 2.0 * length_inner_cube_) {
    const double suggested_value = 2.0 * length_inner_cube_ * sqrt(3.0);
    PARSE_ERROR(
        context,
        "The radius for the enveloping cube is too small! The Frustums will be "
        "malformed. A recommended radius is:\n"
            << suggested_value);
  }
  if (object_A_.outer_radius < object_A_.inner_radius) {
    PARSE_ERROR(context,
                "ObjectA's inner radius must be less than its outer radius.");
  }
  if (object_B_.outer_radius < object_B_.inner_radius) {
    PARSE_ERROR(context,
                "ObjectB's inner radius must be less than its outer radius.");
  }
  if (object_A_.use_logarithmic_map and not object_A_.is_excised()) {
    PARSE_ERROR(
        context,
        "Using a logarithmically spaced radial grid in the part "
        "of Layer 1 enveloping Object A requires excising the interior of "
        "Object A");
  }
  if (object_B_.use_logarithmic_map and not object_B_.is_excised()) {
    PARSE_ERROR(
        context,
        "Using a logarithmically spaced radial grid in the part "
        "of Layer 1 enveloping Object B requires excising the interior of "
        "Object B");
  }
  if (object_A_.is_excised() and
      ((*object_A_.inner_boundary_condition == nullptr) !=
       (outer_boundary_condition_ == nullptr))) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  if (object_B_.is_excised() and
      ((*object_B_.inner_boundary_condition == nullptr) !=
       (outer_boundary_condition_ == nullptr))) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_) or
      (object_A_.is_excised() and
       is_periodic(*object_A_.inner_boundary_condition)) or
      (object_B_.is_excised() and
       is_periodic(*object_B_.inner_boundary_condition))) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  // Create block names and groups
  static std::array<std::string, 6> wedge_directions{
      "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX", "LowerX"};
  const auto add_object_region = [this](const std::string& object_name,
                                        const std::string& region_name) {
    for (const std::string& wedge_direction : wedge_directions) {
      const std::string block_name =
          // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
          object_name + region_name + wedge_direction;
      block_names_.push_back(block_name);
      block_groups_[object_name + region_name].insert(block_name);
    }
  };
  const auto add_object_interior = [this](const std::string& object_name) {
    const std::string block_name = object_name + "Interior";
    block_names_.push_back(block_name);
  };
  const auto add_outer_region = [this](const std::string& region_name) {
    for (const std::string& wedge_direction : wedge_directions) {
      for (const std::string& leftright : {"Left"s, "Right"s}) {
        if ((wedge_direction == "UpperX" and leftright == "Left") or
            (wedge_direction == "LowerX" and leftright == "Right")) {
          // The outer regions are divided in half perpendicular to the
          // x-axis at x=0. Therefore, the left side only has a block in
          // negative x-direction, and the right side only has one in
          // positive x-direction.
          continue;
        }
        // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
        const std::string block_name =
            region_name + wedge_direction +
            (wedge_direction == "UpperX" or wedge_direction == "LowerX"
                 ? ""
                 : leftright);
        block_names_.push_back(block_name);
        block_groups_[region_name].insert(block_name);
      }
    }
  };
  add_object_region("ObjectA", "Shell");  // 6 blocks
  add_object_region("ObjectA", "Cube");   // 6 blocks
  add_object_region("ObjectB", "Shell");  // 6 blocks
  add_object_region("ObjectB", "Cube");   // 6 blocks
  add_outer_region("EnvelopingCube");     // 10 blocks
  if (need_cube_to_sphere_transition_) {
    add_outer_region("CubedShell");  // 10 blocks
  }
  add_outer_region("OuterShell");         // 10 blocks
  if (not object_A_.is_excised()) {
    add_object_interior("ObjectA");  // 1 block
  }
  if (not object_B_.is_excised()) {
    add_object_interior("ObjectB");  // 1 block
  }
  ASSERT(block_names_.size() == number_of_blocks_,
         "Number of block names (" << block_names_.size()
                                   << ") doesn't match number of blocks ("
                                   << number_of_blocks_ << ").");

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

  // Compute the inner radius of the outer spherical shell. The computation
  // makes use of the refinement, so this can't be done earlier.
  if (radius_enveloping_sphere.has_value()) {
    radius_enveloping_sphere_ = radius_enveloping_sphere.value();
    if (radius_enveloping_sphere_ <= radius_enveloping_cube_ or
        radius_enveloping_sphere_ >= outer_radius_domain_) {
      PARSE_ERROR(
          context,
          "The 'OuterShell.InnerRadius' must be within 'EnvelopingCube.Radius' "
          "(" << radius_enveloping_cube_
              << ") and 'OuterShell.OuterRadius' (" << outer_radius_domain_
              << "), but is: " << radius_enveloping_sphere_
              << ". Set it to 'Auto' so a reasonable value is chosen "
                 "automatically.");
    }
  } else if (frustum_sphericity == 1.0) {
    radius_enveloping_sphere_ = radius_enveloping_cube_;
  } else {
    // Adjust the outer boundary of the cubed sphere to conform to the spacing
    // of the spherical shells after refinement, so the cubed sphere is the same
    // size as the first radial division of the spherical shell (for linear
    // mapping) or smaller by the same factor as adjacent radial divisions in
    // the spherical shell (for logarithmic or inverse mapping).
    const size_t addition_to_outer_layer_radial_refinement_level =
        initial_refinement_[44][2] - initial_refinement_[34][2];
    const size_t radial_divisions_in_outer_layers =
        pow(2, addition_to_outer_layer_radial_refinement_level) + 1;
    // Use the `Interval` class to divide the interval between
    // `radius_enveloping_cube_` and `outer_radius_domain_` into
    // `radial_divisions_in_outer_layers` pieces. Choose
    // `radius_enveloping_sphere_` as the first of those pieces.
    radius_enveloping_sphere_ =
        domain::CoordinateMaps::Interval{// Source interval
                                         0., 1.,
                                         // Target interval
                                         radius_enveloping_cube_,
                                         outer_radius_domain_,
                                         // Distribution in target interval
                                         radial_distribution_outer_shell_,
                                         // Position of the singularity for log
                                         // and 1/r maps (in target interval)
                                         0.}(std::array<double, 1>{
            {// Inner radius in source interval that is mapped to target
             // interval
             1. / static_cast<double>(radial_divisions_in_outer_layers)}})[0];
  }
}

// Time-dependent constructor, with additional options for specifying
// the time-dependent maps
BinaryCompactObject::BinaryCompactObject(
    double initial_time, double expansion_map_outer_boundary,
    double initial_expansion, double initial_expansion_velocity,
    double asymptotic_velocity_outer_boundary,
    double decay_timescale_outer_boundary_velocity,
    std::array<double, 3> initial_angular_velocity,
    std::array<double, 2> initial_size_map_values,
    std::array<double, 2> initial_size_map_velocities,
    std::array<double, 2> initial_size_map_accelerations, Object object_A,
    Object object_B, double radius_enveloping_cube, double outer_radius_domain,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    bool use_projective_map, double frustum_sphericity,
    const std::optional<double>& radius_enveloping_sphere,
    CoordinateMaps::Distribution radial_distribution_outer_shell,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : BinaryCompactObject(
          std::move(object_A), std::move(object_B), radius_enveloping_cube,
          outer_radius_domain, initial_refinement,
          initial_number_of_grid_points, use_projective_map, frustum_sphericity,
          radius_enveloping_sphere, radial_distribution_outer_shell,
          std::move(outer_boundary_condition), context) {
  enable_time_dependence_ = true;
  initial_time_ = initial_time;
  expansion_map_outer_boundary_ = expansion_map_outer_boundary;
  initial_expansion_ = initial_expansion;
  initial_expansion_velocity_ = initial_expansion_velocity;
  asymptotic_velocity_outer_boundary_ = asymptotic_velocity_outer_boundary;
  decay_timescale_outer_boundary_velocity_ =
      decay_timescale_outer_boundary_velocity;
  // quat = (cos(theta/2), nhat*sin(theta/2)) but we always take theta = 0
  // initially
  initial_quaternion_ = DataVector{{1.0, 0.0, 0.0, 0.0}};
  initial_size_map_values_ = initial_size_map_values;
  initial_size_map_velocities_ = initial_size_map_velocities;
  initial_size_map_accelerations_ = initial_size_map_accelerations;

  for (size_t i = 0; i < initial_angular_velocity.size(); i++) {
    initial_angular_velocity_[i] = gsl::at(initial_angular_velocity, i);
  }
}

Domain<3> BinaryCompactObject::create_domain() const {
  const double inner_sphericity_A = object_A_.is_excised() ? 1.0 : 0.0;
  const double inner_sphericity_B = object_B_.is_excised() ? 1.0 : 0.0;

  using Maps = std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>;
  using BcMap = DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>;

  std::vector<BcMap> boundary_conditions_all_blocks{};
  // Add an empty map to the boundary conditions for blocks that have no
  // external boundaries, because the `boundary_conditions_all_blocks` expects
  // an entry for every block. This lambda avoid code duplication below.
  const auto add_no_boundary_conditions =
      [this, &boundary_conditions_all_blocks](const Maps& local_maps) {
        if (outer_boundary_condition_ != nullptr) {
          for (size_t i = 0; i < local_maps.size(); ++i) {
            boundary_conditions_all_blocks.emplace_back(BcMap{});
          }
        }
      };

  const std::vector<domain::CoordinateMaps::Distribution>
      object_A_radial_distribution{
          object_A_.use_logarithmic_map
              ? domain::CoordinateMaps::Distribution::Logarithmic
              : domain::CoordinateMaps::Distribution::Linear};

  const std::vector<domain::CoordinateMaps::Distribution>
      object_B_radial_distribution{
          object_B_.use_logarithmic_map
              ? domain::CoordinateMaps::Distribution::Logarithmic
              : domain::CoordinateMaps::Distribution::Linear};

  Maps maps{};

  // --- Blocks enclosing each object (24 blocks) ---
  //
  // Each object is surrounded by 6 inner wedges that make a sphere, and another
  // 6 outer wedges that transition to a cube.

  // ObjectA/B is on the left/right, respectively.
  const Translation translation_A{
      Affine{-1.0, 1.0, -1.0 + object_A_.x_coord, 1.0 + object_A_.x_coord},
      Identity2D{}};
  const Translation translation_B{
      Affine{-1.0, 1.0, -1.0 + object_B_.x_coord, 1.0 + object_B_.x_coord},
      Identity2D{}};

  const auto make_center_maps = [](std::vector<CoordinateMaps::Wedge<3>>
                                       local_wedges,
                                   const Translation& local_translation,
                                   const Object& local_object) {
    if (local_object.shape.has_value()) {
      const auto& kerr_horizon = local_object.shape.value();
      const size_t l_max = kerr_horizon.modes[0];
      const size_t m_max = kerr_horizon.modes[1];
      const YlmSpherepack ylm{l_max, m_max};
      const DataVector radial_distortion =
          1. - get(gr::Solutions::kerr_radius(
                   local_object.inner_radius, ylm.theta_phi_points(),
                   kerr_horizon.mass, kerr_horizon.dimensionless_spin)) /
                   local_object.inner_radius;
      auto radial_distortion_coefs = ylm.phys_to_spec(radial_distortion);
      const domain::CoordinateMaps::TimeDependent::Shape shape_map{
          {{local_object.x_coord, 0., 0.}},
          l_max,
          m_max,
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(local_object.inner_radius,
                                                 local_object.outer_radius),
          std::move(radial_distortion_coefs)};
      return domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                     Frame::Inertial, 3>(
          std::move(local_wedges), local_translation, shape_map);
    } else {
      return domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                     Frame::Inertial, 3>(
          std::move(local_wedges), local_translation);
    }
  };
  Maps maps_center_A = make_center_maps(
      sph_wedge_coordinate_maps(object_A_.inner_radius, object_A_.outer_radius,
                                inner_sphericity_A, 1.0, use_equiangular_map_,
                                false, {}, object_A_radial_distribution),
      translation_A, object_A_);
  Maps maps_cube_A =
      domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                              Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(object_A_.outer_radius,
                                    sqrt(3.0) * 0.5 * length_inner_cube_, 1.0,
                                    0.0, use_equiangular_map_),
          translation_A);
  Maps maps_center_B = make_center_maps(
      sph_wedge_coordinate_maps(object_B_.inner_radius, object_B_.outer_radius,
                                inner_sphericity_B, 1.0, use_equiangular_map_,
                                false, {}, object_B_radial_distribution),
      translation_B, object_B_);
  Maps maps_cube_B =
      domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                              Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(object_B_.outer_radius,
                                    sqrt(3.0) * 0.5 * length_inner_cube_, 1.0,
                                    0.0, use_equiangular_map_),
          translation_B);

  if (outer_boundary_condition_ != nullptr) {
    for (size_t i = 0; i < maps_center_A.size(); ++i) {
      BcMap bcs{};
      if (object_A_.is_excised()) {
        bcs[Direction<3>::lower_zeta()] =
            (*object_A_.inner_boundary_condition)->get_clone();
      }
      boundary_conditions_all_blocks.push_back(std::move(bcs));
    }
  }
  std::move(maps_center_A.begin(), maps_center_A.end(),
            std::back_inserter(maps));
  add_no_boundary_conditions(maps_cube_A);
  std::move(maps_cube_A.begin(), maps_cube_A.end(), std::back_inserter(maps));
  if (outer_boundary_condition_ != nullptr) {
    for (size_t i = 0; i < maps_center_B.size(); ++i) {
      BcMap bcs{};
      if (object_B_.is_excised()) {
        bcs[Direction<3>::lower_zeta()] =
            (*object_B_.inner_boundary_condition)->get_clone();
      }
      boundary_conditions_all_blocks.push_back(std::move(bcs));
    }
  }
  std::move(maps_center_B.begin(), maps_center_B.end(),
            std::back_inserter(maps));
  add_no_boundary_conditions(maps_cube_B);
  std::move(maps_cube_B.begin(), maps_cube_B.end(), std::back_inserter(maps));

  // --- Frustums enclosing both objects (10 blocks) ---
  //
  // The two abutting cubes are enclosed by a layer of frustums that form a cube
  // (if frustum_sphericity_ is 0) or a sphere (if frustum_sphericity_ is 1)
  // surrounding both objects. While the two objects can be offset from the
  // origin to account for their center of mass, the enclosing frustums are
  // centered at the origin.
  Maps maps_frustums = domain::make_vector_coordinate_map_base<
      Frame::BlockLogical, Frame::Inertial, 3>(
      frustum_coordinate_maps(length_inner_cube_, length_outer_cube_,
                              use_equiangular_map_, {{-translation_, 0.0, 0.0}},
                              projective_scale_factor_, frustum_sphericity_));
  add_no_boundary_conditions(maps_frustums);
  std::move(maps_frustums.begin(), maps_frustums.end(),
            std::back_inserter(maps));

  // --- (Optional) transition from frustums to sphere (10 blocks) ---
  //
  // Another layer of wedges transitions from the surrounding frustums to a
  // surrounding sphere. Not needed when the surrounding frustums are already
  // spherical.
  if (need_cube_to_sphere_transition_) {
    Maps maps_first_outer_shell = domain::make_vector_coordinate_map_base<
        Frame::BlockLogical, Frame::Inertial, 3>(sph_wedge_coordinate_maps(
        radius_enveloping_cube_, radius_enveloping_sphere_, frustum_sphericity_,
        1.0, use_equiangular_map_, true, {},
        {domain::CoordinateMaps::Distribution::Linear}));
    add_no_boundary_conditions(maps_first_outer_shell);
    std::move(maps_first_outer_shell.begin(), maps_first_outer_shell.end(),
              std::back_inserter(maps));
  }

  // --- Outer spherical shell (10 blocks) ---
  Maps maps_second_outer_shell = domain::make_vector_coordinate_map_base<
      Frame::BlockLogical, Frame::Inertial, 3>(sph_wedge_coordinate_maps(
      radius_enveloping_sphere_, outer_radius_domain_, 1.0, 1.0,
      use_equiangular_map_, true, {}, {radial_distribution_outer_shell_}));
  if (outer_boundary_condition_ != nullptr) {
    // The outer 10 wedges all have to have the outer boundary condition
    // applied
    for (size_t i = 0; i < maps_second_outer_shell.size(); ++i) {
      BcMap bcs{};
      bcs[Direction<3>::upper_zeta()] = outer_boundary_condition_->get_clone();
      boundary_conditions_all_blocks.push_back(std::move(bcs));
    }
  }
  std::move(maps_second_outer_shell.begin(), maps_second_outer_shell.end(),
            std::back_inserter(maps));

  // --- (Optional) object centers (0 to 2 blocks) ---
  //
  // Each object can optionally be filled with a cube-shaped block, in which
  // case the enclosing wedges configured above transition from the cube to a
  // sphere.
  if (not object_A_.is_excised()) {
    if (outer_boundary_condition_ != nullptr) {
      boundary_conditions_all_blocks.emplace_back(BcMap{});
    }

    const double scaled_r_inner_A = object_A_.inner_radius / sqrt(3.0);
    if (use_equiangular_map_) {
      maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Equiangular3D{Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                        scaled_r_inner_A),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                        scaled_r_inner_A),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                        scaled_r_inner_A)},
              translation_A));
    } else {
      maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Affine3D{
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A)},
              translation_A));
    }
  }
  if (not object_B_.is_excised()) {
    if (outer_boundary_condition_ != nullptr) {
      boundary_conditions_all_blocks.emplace_back(BcMap{});
    }

    const double scaled_r_inner_B = object_B_.inner_radius / sqrt(3.0);
    if (use_equiangular_map_) {
      maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Equiangular3D{Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                        scaled_r_inner_B),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                        scaled_r_inner_B),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                        scaled_r_inner_B)},
              translation_B));
    } else {
      maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Affine3D{
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B)},
              translation_B));
    }
  }

  // Excision spheres
  // - Block 0 through 5 enclose object A, and 12 through 17 enclose object B.
  // - The 3D wedge map is oriented such that the lower-zeta logical direction
  //   points radially inward.
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};
  if (object_A_.is_excised()) {
    excision_spheres.emplace(
        "ObjectAExcisionSphere",
        ExcisionSphere<3>{object_A_.inner_radius,
                          {{object_A_.x_coord, 0.0, 0.0}},
                          {{0, Direction<3>::lower_zeta()},
                           {1, Direction<3>::lower_zeta()},
                           {2, Direction<3>::lower_zeta()},
                           {3, Direction<3>::lower_zeta()},
                           {4, Direction<3>::lower_zeta()},
                           {5, Direction<3>::lower_zeta()}}});
  }
  if (object_B_.is_excised()) {
    excision_spheres.emplace(
        "ObjectBExcisionSphere",
        ExcisionSphere<3>{object_B_.inner_radius,
                          {{object_B_.x_coord, 0.0, 0.0}},
                          {{12, Direction<3>::lower_zeta()},
                           {13, Direction<3>::lower_zeta()},
                           {14, Direction<3>::lower_zeta()},
                           {15, Direction<3>::lower_zeta()},
                           {16, Direction<3>::lower_zeta()},
                           {17, Direction<3>::lower_zeta()}}});
  }

  const size_t num_biradial_layers = need_cube_to_sphere_transition_ ? 3 : 2;
  Domain<3> domain{std::move(maps),
                   corners_for_biradially_layered_domains(
                       2, num_biradial_layers, not object_A_.is_excised(),
                       not object_B_.is_excised()),
                   {},
                   std::move(boundary_conditions_all_blocks),
                   std::move(excision_spheres)};

  // Inject the hard-coded time-dependence
  if (enable_time_dependence_) {
    // Note on frames: Because the relevant maps will all be composed before
    // they are used, all maps here go from Frame::Grid (the frame after the
    // final time-independent map is applied) to Frame::Inertial
    // (the frame after the final time-dependent map is applied).
    using CubicScaleMap = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
    using CubicScaleMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap>;

    using RotationMap3D = domain::CoordinateMaps::TimeDependent::Rotation<3>;
    using RotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotationMap3D>;

    using CubicScaleAndRotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap,
                              RotationMap3D>;

    using CompressionMap =
        domain::CoordinateMaps::TimeDependent::SphericalCompression<false>;
    using CompressionMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap>;

    using CompressionAndCubicScaleAndRotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap,
                              CubicScaleMap, RotationMap3D>;

    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps{number_of_blocks_};

    // Some maps (e.g. expansion, rotation) are applied to all blocks,
    // while other maps (e.g. size) are only applied to some blocks. So
    // there are several different distinct combinations of time-dependent
    // maps that will be applied.
    // Here, set the time-dependent maps for each distinct combination in
    // a single block. Then, set the maps of the other blocks by cloning
    // the maps from the appropriate block.

    // All blocks except possibly blocks 0-5 and 12-17 get the same map, so
    // initialize the final block with the "base" map (here a composition of an
    // expansion and a rotation).
    block_maps[number_of_blocks_ - 1] =
        std::make_unique<CubicScaleAndRotationMapForComposition>(
            domain::push_back(
                CubicScaleMapForComposition{CubicScaleMap{
                    expansion_map_outer_boundary_,
                    expansion_function_of_time_name_,
                    expansion_function_of_time_name_ + "OuterBoundary"s}},
                RotationMapForComposition{
                    RotationMap3D{rotation_function_of_time_name_}}));

    // Initialize the first block of the layer 1 blocks for each object
    // (specifically, initialize block 0 and block 12). If excising interior
    // A or B, the block maps for the coresponding layer 1 blocks (blocks 0-5
    // for object A, blocks 12-17 for object B) should also include a size map.
    // If not excising interior A or B, the layer 1 blocks for that object
    // will have the same map as the final block.
    if (object_A_.is_excised()) {
      block_maps[0] = std::make_unique<
          CompressionAndCubicScaleAndRotationMapForComposition>(
          domain::push_back(
              CompressionMapForComposition{
                  CompressionMap{size_map_function_of_time_names_[0],
                                 object_A_.inner_radius,
                                 object_A_.outer_radius,
                                 {{object_A_.x_coord, 0.0, 0.0}}}},
              domain::push_back(
                  CubicScaleMapForComposition{CubicScaleMap{
                      expansion_map_outer_boundary_,
                      expansion_function_of_time_name_,
                      expansion_function_of_time_name_ + "OuterBoundary"s}},
                  RotationMapForComposition{
                      RotationMap3D{rotation_function_of_time_name_}})));
    } else {
      block_maps[0] = block_maps[number_of_blocks_ - 1]->get_clone();
    }
    if (object_B_.is_excised()) {
      block_maps[12] = std::make_unique<
          CompressionAndCubicScaleAndRotationMapForComposition>(
          domain::push_back(
              CompressionMapForComposition{
                  CompressionMap{size_map_function_of_time_names_[1],
                                 object_B_.inner_radius,
                                 object_B_.outer_radius,
                                 {{object_B_.x_coord, 0.0, 0.0}}}},
              domain::push_back(
                  CubicScaleMapForComposition{CubicScaleMap{
                      expansion_map_outer_boundary_,
                      expansion_function_of_time_name_,
                      expansion_function_of_time_name_ + "OuterBoundary"}},
                  RotationMapForComposition{
                      RotationMap3D{rotation_function_of_time_name_}})));
    } else {
      block_maps[12] = block_maps[number_of_blocks_ - 1]->get_clone();
    }

    // Fill in the rest of the block maps by cloning the relevant maps
    for (size_t block = 1; block < number_of_blocks_ - 1; ++block) {
      if (block < 6) {
        block_maps[block] = block_maps[0]->get_clone();
      } else if (block == 12) {
        continue;  // block 12 already initialized
      } else if (block > 12 and block < 18) {
        block_maps[block] = block_maps[12]->get_clone();
      } else {
        block_maps[block] = block_maps[number_of_blocks_ - 1]->get_clone();
      }
    }

    // Finally, inject the time dependent maps into the corresponding blocks
    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(block,
                                                 std::move(block_maps[block]));
    }
  }

  return domain;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
BinaryCompactObject::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  if (not enable_time_dependence_) {
    return result;
  }

  // Get existing function of time names that are used for the maps and assign
  // their initial expiration time to infinity (i.e. not expiring)
  std::unordered_map<std::string, double> expiration_times{
      {expansion_function_of_time_name_,
       std::numeric_limits<double>::infinity()},
      {rotation_function_of_time_name_,
       std::numeric_limits<double>::infinity()},
      {size_map_function_of_time_names_[0],
       std::numeric_limits<double>::infinity()},
      {size_map_function_of_time_names_[1],
       std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  // ExpansionMap FunctionOfTime for the function \f$a(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {{initial_expansion_}, {initial_expansion_velocity_}, {0.0}}},
          expiration_times.at(expansion_function_of_time_name_));

  // ExpansionMap FunctionOfTime for the function \f$b(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_function_of_time_name_ + "OuterBoundary"s] =
      std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
          1.0, initial_time_, asymptotic_velocity_outer_boundary_,
          decay_timescale_outer_boundary_velocity_);

  // RotationMap FunctionOfTime for the rotation angles about each axis.
  // The initial rotation angles don't matter as we never actually use the
  // angles themselves. We only use their derivatives (omega) to determine map
  // parameters. In theory we could determine each initital angle from the input
  // axis-angle representation, but we don't need to.
  result[rotation_function_of_time_name_] =
      std::make_unique<FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          initial_time_, std::array<DataVector, 1>{initial_quaternion_},
          std::array<DataVector, 4>{
              {{3, 0.0}, initial_angular_velocity_, {3, 0.0}, {3, 0.0}}},
          expiration_times.at(rotation_function_of_time_name_));

  // CompressionMap FunctionOfTime for objects A and B
  for (size_t i = 0; i < size_map_function_of_time_names_.size(); i++) {
    result[gsl::at(size_map_function_of_time_names_, i)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time_,
            std::array<DataVector, 4>{
                {{gsl::at(initial_size_map_values_, i)},
                 {gsl::at(initial_size_map_velocities_, i)},
                 {gsl::at(initial_size_map_accelerations_, i)},
                 {0.0}}},
            expiration_times.at(gsl::at(size_map_function_of_time_names_, i)));
  }

  return result;
}
}  // namespace domain::creators
