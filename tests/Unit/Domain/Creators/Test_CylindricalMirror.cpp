// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/CylindricalMirror.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {

void test_cylindrical_mirror_boundaries(const bool use_equiangular_map) {
  INFO("Cylindrical Mirror boundaries");
  using block_specs = std::vector<std::vector<size_t>>;
  // definition of an arbitrary cylindrical mirror
  const std::vector<double> partition_along_radius = {2.0, 4.0, 6.0};
  const std::vector<double> partition_along_z = {-2.5, 1.25, 5.0};
  const std::vector<size_t> refinement_level_of_square = {0, 1};
  const block_specs refinement_level_along_radius = {{0, 1}, {1, 2}};
  const block_specs refinement_level_along_angle = {{1, 0}, {2, 1}};
  const block_specs refinement_level_along_z = {{1, 2, 3}, {2, 1, 0}};
  const std::vector<std::array<size_t, 2>> grid_points_of_square = {{2, 3},
                                                                    {4, 5}};
  const block_specs grid_points_along_radius = {{6, 7}, {8, 9}};
  const block_specs grid_points_along_angle = {{2, 4}, {6, 8}};
  const block_specs grid_points_along_z = {{2, 3, 4}, {5, 6, 7}};

  const std::vector<std::array<size_t, 3>> expected_refinement_level{
      {0, 0, 1}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {1, 0, 3},
      {1, 0, 3}, {1, 0, 3}, {1, 0, 3}, {1, 1, 2}, {1, 2, 1}, {1, 2, 1},
      {1, 2, 1}, {1, 2, 1}, {2, 1, 0}, {2, 1, 0}, {2, 1, 0}, {2, 1, 0}};

  const std::vector<std::array<size_t, 3>>& expected_extents{
      {2, 3, 2}, {6, 2, 3}, {6, 2, 3}, {6, 2, 3}, {6, 2, 3}, {7, 4, 4},
      {7, 4, 4}, {7, 4, 4}, {7, 4, 4}, {4, 5, 5}, {8, 6, 6}, {8, 6, 6},
      {8, 6, 6}, {8, 6, 6}, {9, 8, 7}, {9, 8, 7}, {9, 8, 7}, {9, 8, 7}};

  const creators::CylindricalMirror cylindrical_mirror{
      partition_along_radius,   refinement_level_of_square,
      grid_points_of_square,    refinement_level_along_radius,
      grid_points_along_radius, refinement_level_along_angle,
      grid_points_along_angle,  partition_along_z,
      refinement_level_along_z, grid_points_along_z,
      use_equiangular_map};
  test_physical_separation(cylindrical_mirror.create_domain().blocks());

  const auto domain = cylindrical_mirror.create_domain();
  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}});
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{};
  std::vector<std::unordered_set<Direction<3>>> expected_external_boundaries{};
  using TargetFrame = Frame::Inertial;
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
      coord_maps{};
  expected_block_neighbors = std::vector<DirectionMap<3, BlockNeighbor<3>>>{
      // Block 0 - layer 0 center 0
      {{Direction<3>::upper_xi(), {1, aligned_orientation}},
       {Direction<3>::upper_eta(), {2, quarter_turn_ccw}},
       {Direction<3>::lower_xi(), {3, half_turn}},
       {Direction<3>::lower_eta(), {4, quarter_turn_cw}},
       {Direction<3>::upper_zeta(), {0 + 9, aligned_orientation}}},
      // Block 1 - layer 0 east 1
      {{Direction<3>::lower_eta(), {4, aligned_orientation}},
       {Direction<3>::upper_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, aligned_orientation}},
       {Direction<3>::upper_xi(), {1 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {1 + 9, aligned_orientation}}},
      // Block 2 - layer 0 north 2
      {{Direction<3>::lower_eta(), {1, aligned_orientation}},
       {Direction<3>::upper_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, quarter_turn_cw}},
       {Direction<3>::upper_xi(), {2 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {2 + 9, aligned_orientation}}},
      // Block 3 - layer 0 west 3
      {{Direction<3>::lower_eta(), {2, aligned_orientation}},
       {Direction<3>::upper_eta(), {4, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, half_turn}},
       {Direction<3>::upper_xi(), {3 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {3 + 9, aligned_orientation}}},
      // Block 4 - layer 0 south 4
      {{Direction<3>::lower_eta(), {3, aligned_orientation}},
       {Direction<3>::upper_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, quarter_turn_ccw}},
       {Direction<3>::upper_xi(), {4 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {4 + 9, aligned_orientation}}},
      // Block 5 - layer 0 east 5
      {{Direction<3>::lower_eta(), {8, aligned_orientation}},
       {Direction<3>::upper_eta(), {6, aligned_orientation}},
       {Direction<3>::lower_xi(), {5 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {5 + 9, aligned_orientation}}},
      // Block 6 - layer 0 north 6
      {{Direction<3>::lower_eta(), {5, aligned_orientation}},
       {Direction<3>::upper_eta(), {7, aligned_orientation}},
       {Direction<3>::lower_xi(), {6 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {6 + 9, aligned_orientation}}},
      // Block 7 - layer 0 west 7
      {{Direction<3>::lower_eta(), {6, aligned_orientation}},
       {Direction<3>::upper_eta(), {8, aligned_orientation}},
       {Direction<3>::lower_xi(), {7 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {7 + 9, aligned_orientation}}},
      // Block 8 - layer 0 south 8
      {{Direction<3>::lower_eta(), {7, aligned_orientation}},
       {Direction<3>::upper_eta(), {5, aligned_orientation}},
       {Direction<3>::lower_xi(), {8 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {8 + 9, aligned_orientation}}},
      // Block 9 - layer 1 center 0
      {{Direction<3>::upper_xi(), {1 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {2 + 9, quarter_turn_ccw}},
       {Direction<3>::lower_xi(), {3 + 9, half_turn}},
       {Direction<3>::lower_eta(), {4 + 9, quarter_turn_cw}},
       {Direction<3>::lower_zeta(), {9 - 9, aligned_orientation}}},
      // Block 10 - layer 1 east 1
      {{Direction<3>::lower_eta(), {4 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {2 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, aligned_orientation}},
       {Direction<3>::upper_xi(), {1 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {10 - 9, aligned_orientation}}},
      // Block 11 - layer 1 north 2
      {{Direction<3>::lower_eta(), {1 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {3 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, quarter_turn_cw}},
       {Direction<3>::upper_xi(), {2 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {11 - 9, aligned_orientation}}},
      // Block 12 - layer 1 west 3
      {{Direction<3>::lower_eta(), {2 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {4 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, half_turn}},
       {Direction<3>::upper_xi(), {3 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {12 - 9, aligned_orientation}}},
      // Block 13 - layer 1 south 4
      {{Direction<3>::lower_eta(), {3 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {1 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, quarter_turn_ccw}},
       {Direction<3>::upper_xi(), {4 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {13 - 9, aligned_orientation}}},
      // Block 14 - layer 1 east 5
      {{Direction<3>::lower_eta(), {8 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {6 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {5 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {14 - 9, aligned_orientation}}},
      // Block 15 - layer 1 north 6
      {{Direction<3>::lower_eta(), {5 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {7 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {6 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {15 - 9, aligned_orientation}}},
      // Block 16 - layer 1 west 7
      {{Direction<3>::lower_eta(), {6 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {8 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {7 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {16 - 9, aligned_orientation}}},
      // Block 17 - layer 1 south 8
      {{Direction<3>::lower_eta(), {7 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {5 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {8 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {17 - 9, aligned_orientation}}}};

  expected_external_boundaries = std::vector<std::unordered_set<Direction<3>>>{
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {{Direction<3>::upper_xi(), Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_xi(), Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_xi(), Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_xi(), Direction<3>::lower_zeta()}},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {{Direction<3>::upper_xi(), Direction<3>::upper_zeta()}},
      {{Direction<3>::upper_xi(), Direction<3>::upper_zeta()}},
      {{Direction<3>::upper_xi(), Direction<3>::upper_zeta()}},
      {{Direction<3>::upper_xi(), Direction<3>::upper_zeta()}}};

  CHECK(cylindrical_mirror.initial_extents() == expected_extents);
  CHECK(cylindrical_mirror.initial_refinement_levels() ==
        expected_refinement_level);
  using TargetFrame = Frame::Inertial;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3DPrism =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
  using Wedge2D = CoordinateMaps::Wedge2D;
  using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;
  if (use_equiangular_map) {
    coord_maps.emplace_back(make_coordinate_map_base<
                            Frame::Logical, TargetFrame>(Equiangular3DPrism{
        Equiangular(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                    partition_along_radius.at(0) / sqrt(2.0)),
        Equiangular(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                    partition_along_radius.at(0) / sqrt(2.0)),
        Affine{-1.0, 1.0, partition_along_z.at(0), partition_along_z.at(1)}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Affine3D{
            Affine(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                   partition_along_radius.at(0) / sqrt(2.0)),
            Affine(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                   partition_along_radius.at(0) / sqrt(2.0)),
            Affine{-1.0, 1.0, partition_along_z.at(0),
                   partition_along_z.at(1)}}));
  }
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(0),
                 partition_along_z.at(1)}}));
  if (use_equiangular_map) {
    coord_maps.emplace_back(make_coordinate_map_base<
                            Frame::Logical, TargetFrame>(Equiangular3DPrism{
        Equiangular(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                    partition_along_radius.at(0) / sqrt(2.0)),
        Equiangular(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                    partition_along_radius.at(0) / sqrt(2.0)),
        Affine{-1.0, 1.0, partition_along_z.at(1), partition_along_z.at(2)}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Affine3D{
            Affine(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                   partition_along_radius.at(0) / sqrt(2.0)),
            Affine(-1.0, 1.0, -1.0 * partition_along_radius.at(0) / sqrt(2.0),
                   partition_along_radius.at(0) / sqrt(2.0)),
            Affine{-1.0, 1.0, partition_along_z.at(1),
                   partition_along_z.at(2)}}));
  }
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(0), partition_along_radius.at(1),
                  0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{partition_along_radius.at(1), partition_along_radius.at(2),
                  1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, partition_along_z.at(1),
                 partition_along_z.at(2)}}));

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);

  test_initial_domain(domain, cylindrical_mirror.initial_refinement_levels());

  Parallel::register_classes_in_list<
      typename creators::CylindricalMirror::maps_list>();
  test_serialization(domain);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.CylindricalMirror", "[Domain][Unit]") {
  test_cylindrical_mirror_boundaries(true);
  test_cylindrical_mirror_boundaries(false);
}
}  // namespace domain
