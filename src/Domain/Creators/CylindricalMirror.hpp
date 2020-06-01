// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
class Wedge2D;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond
namespace domain {
namespace creators {
/// Create a 3D Domain in the shape of a cylinder where the cross-section of a
/// layer is a square surrounded by subsequent shells of four two-dimensional
/// wedges (see Wedge2D).
///
/// \image html Cylinder.png "The Cylinder Domain."
class CylindricalMirror : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge2D,
                                         CoordinateMaps::Affine>>>;

  /// A sorted array with neighbouring elements defining the lower and upper
  /// boundary of subsequent shells.
  struct PartitionInRadius {
    using type = std::vector<double>;
    static constexpr OptionString help = {
        "Half diagonal of inner square and radius of outer shells."};
  };

  struct RefLevSquare {
    using type = std::vector<size_t>;
    static constexpr OptionString help = {
        "Initial refinement in x and y for the square."};
  };

  struct GridPointsSquare {
    using type = std::vector<std::array<size_t, 2>>;
    static constexpr OptionString help = {
        "Initial # of grid points in [x, y] for the square."};
  };

  struct RefLevRadius {
    using type = std::vector<std::vector<size_t>>;
    static constexpr OptionString help = {
        "Initial refinement along radius for each shell."};
  };

  struct GridPointsRadius {
    using type = std::vector<std::vector<size_t>>;
    static constexpr OptionString help = {
        "Initial # of grid points along radius for each shell."};
  };

  /// Initial refinement level in angular direction for all shells. The total
  /// number of elements in theta is \f$ 4*2^{N} \f$.
  struct RefLevTheta {
    using type = std::vector<std::vector<size_t>>;
    static constexpr OptionString help = {
        "Refinement level along angle for all shells."};
  };

  struct GridPointsTheta {
    using type = std::vector<std::vector<size_t>>;
    static constexpr OptionString help = {
        "Initial # of grid points along angle for all shells."};
  };

  /// A sorted array with neighbouring elements defining the upper and lower
  /// boundary of connected layers.
  struct PartitionInZ {
    using type = std::vector<double>;
    static constexpr OptionString help = {"Position of Layer boundaries."};
  };

  struct RefLevZ {
    using type = std::vector<std::vector<size_t>>;
    static constexpr OptionString help = {"Initial refinement for each layer."};
  };

  struct GridPointsZ {
    using type = std::vector<std::vector<size_t>>;
    static constexpr OptionString help = {
        "Initial # of grid points for each layer."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr OptionString help = {
        "Use equiangular instead of equidistant coordinates."};
    static type default_value() noexcept { return false; }
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr OptionString help = {
        "The time dependence of the moving mesh domain."};
    static type default_value() noexcept;
  };
  using options = tmpl::list<PartitionInRadius, RefLevSquare, GridPointsSquare,
                             RefLevRadius, GridPointsRadius, RefLevTheta,
                             GridPointsTheta, PartitionInZ, RefLevZ,
                             GridPointsZ, UseEquiangularMap, TimeDependence>;

  static constexpr OptionString help{
      "Creates a right circular Cylinder with shells in the radial direction "
      "and layers along its length. The number of Blocks is \f$ (1 + 4*n_s) * "
      "n_z \f$, where \f$n_s\f$ is the number of shells and \f$n_z\f$ the "
      "number of layers. The circularity of the coordinate system changes from "
      "0 to 1 within the first shell. The refinement level and number of grid "
      "points is given for the xy-plane and the z direction separately. In the "
      "cross-section, the former are given as pairs of scalars for the square, "
      "a scalar in the angular direction for all shells and as an array in the "
      "radial direction for all blocks within a shell. Equiangular coordinates "
      "give better gridpoint spacings in the angular direction, while "
      "equidistant coordinates give better gridpoint spacings in the center "
      "block. This Domain uses equidistant coordinates by default. The "
      "boundary conditions are periodic along the cylindrical z-axis by "
      "default."};

  CylindricalMirror(
      typename PartitionInRadius::type partition_along_radius,
      typename RefLevSquare::type refinement_level_of_square,
      typename GridPointsSquare::type grid_points_of_square,
      typename RefLevRadius::type refinement_level_along_radius,
      typename GridPointsRadius::type grid_points_along_radius,
      typename RefLevTheta::type refinement_level_along_angle,
      typename GridPointsTheta::type grid_points_of_shells_along_angle,
      typename PartitionInZ::type partition_along_z,
      typename RefLevZ::type refinement_level_along_z,
      typename GridPointsZ::type grid_points_along_z,
      typename UseEquiangularMap::type use_equiangular_map,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr) noexcept;

  CylindricalMirror() = default;
  CylindricalMirror(const CylindricalMirror&) = delete;
  CylindricalMirror(CylindricalMirror&&) noexcept = default;
  CylindricalMirror& operator=(const CylindricalMirror&) = delete;
  CylindricalMirror& operator=(CylindricalMirror&&) noexcept = default;
  ~CylindricalMirror() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  typename PartitionInRadius::type partition_along_radius_{};
  typename RefLevSquare::type refinement_level_of_square_{};
  typename GridPointsSquare::type grid_points_of_square_{};
  typename RefLevRadius::type refinement_level_along_radius_{};
  typename GridPointsRadius::type grid_points_along_radius_{};
  typename RefLevTheta::type refinement_level_along_angle_{};
  typename GridPointsTheta::type grid_points_along_angle_{};
  typename PartitionInZ::type partition_along_z_{};
  typename RefLevZ::type refinement_level_along_z_{};
  typename GridPointsZ::type grid_points_along_z_{};
  typename UseEquiangularMap::type use_equiangular_map_{false};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
};
}  // namespace creators
}  // namespace domain
