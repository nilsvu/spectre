// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CylindricalMirror.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Logical;
struct Inertial;
}  // namespace Frame

namespace domain {
namespace creators {

template <typename data_type>
data_type elementwise_addition(const data_type first_array,
                               const data_type second_array) noexcept {
  ASSERT(first_array.size() == second_array.size(),
         "The array sizes are a:"
             << first_array.size() << " and b: " << second_array.size()
             << ", but must be equal for elementwise addition.");
  data_type result{};
  for (size_t i = 0; i < first_array.size(); i++) {
    result.at(i) = first_array.at(i) + second_array.at(i);
  }
  return result;
}

std::vector<std::array<size_t, 8>> list_corners(
    const std::vector<double> partition_along_r,
    const std::vector<double> partition_along_z) noexcept {
  using block_corners = std::array<size_t, 8>;
  std::vector<block_corners> corner_list;
  const size_t n_l = 4 * partition_along_r.size();  // corners per layer
  const block_corners center{{0, 1, 2, 3, n_l + 0, n_l + 1, n_l + 2, n_l + 3}};
  const block_corners east{{1, 5, 3, 7, n_l + 1, n_l + 5, n_l + 3, n_l + 7}};
  const block_corners north{{3, 7, 2, 6, n_l + 3, n_l + 7, n_l + 2, n_l + 6}};
  const block_corners west{{2, 6, 0, 4, n_l + 2, n_l + 6, n_l + 0, n_l + 4}};
  const block_corners south{{0, 4, 1, 5, n_l + 0, n_l + 4, n_l + 1, n_l + 5}};
  for (size_t i = 0; i < partition_along_z.size() - 1; i++) {
    corner_list.push_back(elementwise_addition(center, make_array<8>(i * n_l)));
    for (size_t j = 0; j < partition_along_r.size() - 1; j++) {
      auto offset = make_array<8>(i * n_l + 4 * j);
      corner_list.push_back(elementwise_addition(east, offset));   //+x wedge
      corner_list.push_back(elementwise_addition(north, offset));  //+y wedge
      corner_list.push_back(elementwise_addition(west, offset));   //-x wedge
      corner_list.push_back(elementwise_addition(south, offset));  //-y wedge
    }
  }
  return corner_list;
}

std::vector<
    std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
list_blocks(const std::vector<double> partition_along_r,
            const std::vector<double> partition_along_z,
            const bool use_equiangular_map) noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3DPrism =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
  using Wedge2D = CoordinateMaps::Wedge2D;
  using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      concatenate_blocks;
  double circularity_inner;
  for (size_t i = 1; i < partition_along_z.size(); i++) {
    if (use_equiangular_map) {
      concatenate_blocks.push_back(
          make_coordinate_map_base<Frame::Logical,
                                   Frame::Inertial>(Equiangular3DPrism{
              Equiangular(-1.0, 1.0, -1.0 * partition_along_r.at(0) / sqrt(2.0),
                          partition_along_r.at(0) / sqrt(2.0)),
              Equiangular(-1.0, 1.0, -1.0 * partition_along_r.at(0) / sqrt(2.0),
                          partition_along_r.at(0) / sqrt(2.0)),
              Affine{-1.0, 1.0, partition_along_z.at(i - 1),
                     partition_along_z.at(i)}}));
    } else {
      concatenate_blocks.push_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(Affine3D{
              Affine(-1.0, 1.0, -1.0 * partition_along_r.at(0) / sqrt(2.0),
                     partition_along_r.at(0) / sqrt(2.0)),
              Affine(-1.0, 1.0, -1.0 * partition_along_r.at(0) / sqrt(2.0),
                     partition_along_r.at(0) / sqrt(2.0)),
              Affine{-1.0, 1.0, partition_along_z.at(i - 1),
                     partition_along_z.at(i)}}));
    }
    circularity_inner = 0.;
    for (size_t j = 1; j < partition_along_r.size(); j++) {
      concatenate_blocks.push_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Wedge3DPrism{
                  Wedge2D{partition_along_r.at(j - 1), partition_along_r.at(j),
                          circularity_inner, 1.0,
                          OrientationMap<2>{std::array<Direction<2>, 2>{
                              {Direction<2>::upper_xi(),
                               Direction<2>::upper_eta()}}},
                          use_equiangular_map},
                  Affine{-1.0, 1.0, partition_along_z.at(i - 1),
                         partition_along_z.at(i)}}));
      concatenate_blocks.push_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Wedge3DPrism{
                  Wedge2D{partition_along_r.at(j - 1), partition_along_r.at(j),
                          circularity_inner, 1.0,
                          OrientationMap<2>{std::array<Direction<2>, 2>{
                              {Direction<2>::lower_eta(),
                               Direction<2>::upper_xi()}}},
                          use_equiangular_map},
                  Affine{-1.0, 1.0, partition_along_z.at(i - 1),
                         partition_along_z.at(i)}}));
      concatenate_blocks.push_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Wedge3DPrism{
                  Wedge2D{partition_along_r.at(j - 1), partition_along_r.at(j),
                          circularity_inner, 1.0,
                          OrientationMap<2>{std::array<Direction<2>, 2>{
                              {Direction<2>::lower_xi(),
                               Direction<2>::lower_eta()}}},
                          use_equiangular_map},
                  Affine{-1.0, 1.0, partition_along_z.at(i - 1),
                         partition_along_z.at(i)}}));
      concatenate_blocks.push_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Wedge3DPrism{
                  Wedge2D{partition_along_r.at(j - 1), partition_along_r.at(j),
                          circularity_inner, 1.0,
                          OrientationMap<2>{std::array<Direction<2>, 2>{
                              {Direction<2>::upper_eta(),
                               Direction<2>::lower_xi()}}},
                          use_equiangular_map},
                  Affine{-1.0, 1.0, partition_along_z.at(i - 1),
                         partition_along_z.at(i)}}));
      circularity_inner = 1.;
    }
  }
  return concatenate_blocks;
}

std::vector<std::array<size_t, 3>> list_refinement_levels(
    const std::vector<size_t> refinement_level_of_square,
    const std::vector<std::vector<size_t>> refinement_radius,
    const std::vector<std::vector<size_t>> refinement_angle,
    const std::vector<std::vector<size_t>> refinement_z) noexcept {
  std::vector<std::array<size_t, 3>> refinement_vector;
  for (size_t i = 0; i < refinement_radius.size(); i++) {
    refinement_vector.push_back(
        {{refinement_level_of_square.at(i), refinement_level_of_square.at(i),
          refinement_z.at(i).at(0)}});
    for (size_t j = 0; j < refinement_radius.at(i).size(); j++) {
      std::array<size_t, 3> shell_wedge{{refinement_radius.at(i).at(j),
                                         refinement_angle.at(i).at(j),
                                         refinement_z.at(i).at(j + 1)}};
      for (size_t k = 0; k < 4; k++) {
        refinement_vector.push_back(shell_wedge);
      }
    }
  }
  return refinement_vector;
}

std::vector<std::array<size_t, 3>> list_gridpoints(
    std::vector<std::array<size_t, 2>> gridpoints_of_square,
    std::vector<std::vector<size_t>> gridpoints_of_shells_along_r,
    std::vector<std::vector<size_t>> gridpoints_of_shells_along_theta,
    std::vector<std::vector<size_t>> gridpoints_along_z) noexcept {
  std::vector<std::array<size_t, 3>> gridpoints_vector;
  for (size_t i = 0; i < gridpoints_of_shells_along_r.size(); i++) {
    gridpoints_vector.push_back(
        {{gridpoints_of_square.at(i).at(0), gridpoints_of_square.at(i).at(1),
          gridpoints_along_z.at(i).at(0)}});
    for (size_t j = 0; j < gridpoints_of_shells_along_r.at(i).size(); j++) {
      std::array<size_t, 3> shell_gridpoints{
          {gridpoints_of_shells_along_r.at(i).at(j),
           gridpoints_of_shells_along_theta.at(i).at(j),
           gridpoints_along_z.at(i).at(j + 1)}};
      for (size_t k = 0; k < 4; k++) {
        gridpoints_vector.push_back(shell_gridpoints);
      }
    }
  }
  return gridpoints_vector;
}

CylindricalMirror::CylindricalMirror(
    typename PartitionInRadius::type partition_along_radius,
    typename RefLevSquare::type refinement_level_of_square,
    typename GridPointsSquare::type grid_points_of_square,
    typename RefLevRadius::type refinement_level_along_radius,
    typename GridPointsRadius::type grid_points_along_radius,
    typename RefLevTheta::type refinement_level_along_angle,
    typename GridPointsTheta::type grid_points_along_angle,
    typename PartitionInZ::type partition_along_z,
    typename RefLevZ::type refinement_level_along_z,
    typename GridPointsZ::type grid_points_along_z,
    typename UseEquiangularMap::type use_equiangular_map,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence) noexcept
    // clang-tidy: trivially copyable
    : partition_along_radius_(std::move(partition_along_radius)),  // NOLINT
      refinement_level_of_square_(
          std::move(refinement_level_of_square)),                // NOLINT
      grid_points_of_square_(std::move(grid_points_of_square)),  // NOLINT
      refinement_level_along_radius_(
          std::move(refinement_level_along_radius)),                   // NOLINT
      grid_points_along_radius_(std::move(grid_points_along_radius)),  // NOLINT
      refinement_level_along_angle_(
          std::move(refinement_level_along_angle)),                    // NOLINT
      grid_points_along_angle_(std::move(grid_points_along_angle)),    // NOLINT
      partition_along_z_(std::move(partition_along_z)),                // NOLINT
      refinement_level_along_z_(std::move(refinement_level_along_z)),  // NOLINT
      grid_points_along_z_(std::move(grid_points_along_z)),            // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),            // NOLINT
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
}

Domain<3> CylindricalMirror::create_domain() const noexcept {
  Domain<3> domain{list_blocks(partition_along_radius_, partition_along_z_,
                               use_equiangular_map_),
                   list_corners(partition_along_radius_, partition_along_z_),
                   std::vector<PairOfFaces>{}};
  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps(1)[0]));
  }
  return domain;
}

std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
CylindricalMirror::TimeDependence::default_value() noexcept {
  return std::make_unique<domain::creators::time_dependence::None<3>>();
}

std::vector<std::array<size_t, 3>> CylindricalMirror::initial_extents() const
    noexcept {
  return list_gridpoints(grid_points_of_square_, grid_points_along_radius_,
                         grid_points_along_angle_, grid_points_along_z_);
}

std::vector<std::array<size_t, 3>>
CylindricalMirror::initial_refinement_levels() const noexcept {
  return list_refinement_levels(
      refinement_level_of_square_, refinement_level_along_radius_,
      refinement_level_along_angle_, refinement_level_along_z_);
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CylindricalMirror::functions_of_time() const noexcept {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}

}  // namespace creators
}  // namespace domain
