// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace domain {
namespace {
using Affine = CoordinateMaps::Affine;
using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Translation2D = CoordinateMaps::TimeDependent::Translation<2>;

template <typename... FuncsOfTime>
void test_rectangle_construction(
    const creators::Rectangle& rectangle,
    const std::array<double, 2>& lower_bound,
    const std::array<double, 2>& upper_bound,
    const std::vector<std::array<size_t, 2>>& expected_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const std::vector<DirectionMap<2, BlockNeighbor<2>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<2>>>&
        expected_external_boundaries,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::Grid, Frame::Inertial, 2>>>& expected_grid_to_inertial_maps = {},
    const bool expect_boundary_conditions = false,
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      rectangle, expect_boundary_conditions);
  CHECK(rectangle.grid_anchors().empty());

  CHECK(rectangle.initial_extents() == expected_extents);
  CHECK(rectangle.initial_refinement_levels() == expected_refinement_level);

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(make_coordinate_map_base<
                  Frame::BlockLogical,
                  tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                      Frame::Inertial, Frame::Grid>>(
          Affine2D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]}})),
      10.0, rectangle.functions_of_time(), expected_grid_to_inertial_maps);
  TestHelpers::domain::creators::test_functions_of_time(
      rectangle, expected_functions_of_time, initial_expiration_times);
}

void test_rectangle() {
  INFO("Rectangle");
  const std::vector<std::array<size_t, 2>> grid_points{{{4, 6}}};
  const std::vector<std::array<size_t, 2>> refinement_level{{{3, 2}}};
  const std::array<double, 2> lower_bound{{-1.2, 3.0}};
  const std::array<double, 2> upper_bound{{0.8, 5.0}};
  // default OrientationMap is aligned
  const OrientationMap<2> aligned_orientation{};

  {
    INFO("Rectangle, non-periodic no boundary conditions");
    test_rectangle_construction(
        {lower_bound, upper_bound, refinement_level[0], grid_points[0],
         std::array<bool, 2>{{false, false}}},
        lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{{}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_xi()},
             {Direction<2>::upper_xi()},
             {Direction<2>::lower_eta()},
             {Direction<2>::upper_eta()}}});
  }

  {
    INFO("Rectangle, non-periodic with boundary conditions");
    std::vector<DirectionMap<
        2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        expected_boundary_conditions{1};
    for (const auto& direction : Direction<2>::all_directions()) {
      expected_boundary_conditions[0][direction] = std::make_unique<
          TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<2>>(
          Direction<2>::lower_xi(), 0);
    }
    test_rectangle_construction(
        {lower_bound, upper_bound, refinement_level[0], grid_points[0],
         std::make_unique<
             TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<2>>(
             Direction<2>::lower_xi(), 0),
         nullptr},
        lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{{}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_xi()},
             {Direction<2>::upper_xi()},
             {Direction<2>::lower_eta()},
             {Direction<2>::upper_eta()}}},
        {}, {}, true);
  }

  {
    INFO("Rectangle, periodic in x no boundary conditions");
    test_rectangle_construction(
        {lower_bound, upper_bound, refinement_level[0], grid_points[0],
         std::array<bool, 2>{{true, false}}},
        lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::lower_xi(), {0, aligned_orientation}},
             {Direction<2>::upper_xi(), {0, aligned_orientation}}}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_eta()}, {Direction<2>::upper_eta()}}});
  }

  {
    INFO("Rectangle, periodic in y no boundary conditions");
    test_rectangle_construction(
        {lower_bound, upper_bound, refinement_level[0], grid_points[0],
         std::array<bool, 2>{{false, true}}},
        lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::lower_eta(), {0, aligned_orientation}},
             {Direction<2>::upper_eta(), {0, aligned_orientation}}}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_xi()}, {Direction<2>::upper_xi()}}});
  }

  {
    INFO("Rectangle, periodic in xy no boundary conditions");
    test_rectangle_construction(
        {lower_bound, upper_bound, refinement_level[0], grid_points[0],
         std::array<bool, 2>{{true, true}}},
        lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::lower_xi(), {0, aligned_orientation}},
             {Direction<2>::upper_xi(), {0, aligned_orientation}},
             {Direction<2>::lower_eta(), {0, aligned_orientation}},
             {Direction<2>::upper_eta(), {0, aligned_orientation}}}},
        std::vector<std::unordered_set<Direction<2>>>{{}});
  }

  {
    INFO("Rectangle, periodic in xy with boundary conditions");
    test_rectangle_construction(
        {lower_bound, upper_bound, refinement_level[0], grid_points[0],
         TestHelpers::domain::BoundaryConditions::TestPeriodicBoundaryCondition<
             2>{}
             .get_clone(),
         nullptr},
        lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::lower_xi(), {0, aligned_orientation}},
             {Direction<2>::upper_xi(), {0, aligned_orientation}},
             {Direction<2>::lower_eta(), {0, aligned_orientation}},
             {Direction<2>::upper_eta(), {0, aligned_orientation}}}},
        std::vector<std::unordered_set<Direction<2>>>{{}}, {}, {}, true);
  }
  CHECK_THROWS_WITH(
      creators::Rectangle(
          lower_bound, upper_bound, refinement_level[0], grid_points[0],
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that."));
}

void test_rectangle_factory() {
  // For non-periodic domains:
  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      expected_boundary_conditions{1};
  for (const auto& direction : Direction<2>::all_directions()) {
    expected_boundary_conditions[0][direction] = std::make_unique<
        TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<2>>(
        Direction<2>::lower_xi(), 0);
  }
  const std::vector<std::unordered_set<Direction<2>>>
      expected_external_boundaries{
          {Direction<2>::lower_xi(), Direction<2>::upper_xi(),
           Direction<2>::lower_eta(), Direction<2>::upper_eta()}};

  // for periodic domains:
  const std::vector<DirectionMap<2, BlockNeighbor<2>>> expected_neighbors{
      {{Direction<2>::lower_xi(), {0, {}}},
       {Direction<2>::upper_xi(), {0, {}}}}};

  {
    INFO("Rectangle factory time independent, no boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<2>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                2, domain::creators::Rectangle>>(
        "Rectangle:\n"
        "  LowerBound: [0,0]\n"
        "  UpperBound: [1,2]\n"
        "  IsPeriodicIn: [True,False]\n"
        "  InitialGridPoints: [3,4]\n"
        "  InitialRefinement: [2,3]\n"
        "  TimeDependence: None\n");
    const auto* rectangle_creator =
        dynamic_cast<const creators::Rectangle*>(domain_creator.get());
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        expected_neighbors,
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_eta()}, {Direction<2>::upper_eta()}}});
  }
  {
    INFO("Rectangle factory time independent, with boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<2>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<
                2, domain::creators::Rectangle>>(
        "Rectangle:\n"
        "  LowerBound: [0,0]\n"
        "  UpperBound: [1,2]\n"
        "  InitialGridPoints: [3,4]\n"
        "  InitialRefinement: [2,3]\n"
        "  TimeDependence: None\n"
        "  BoundaryCondition:\n"
        "    TestBoundaryCondition:\n"
        "      Direction: lower-xi\n"
        "      BlockId: 0\n");
    const auto* rectangle_creator =
        dynamic_cast<const creators::Rectangle*>(domain_creator.get());
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        {{}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_xi(), Direction<2>::upper_xi(),
              Direction<2>::lower_eta(), Direction<2>::upper_eta()}}},
        {}, {}, true);
  }
  {
    INFO("Rectangle factory time dependent");
    const auto domain_creator =
        TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<2>,
                                     TestHelpers::domain::BoundaryConditions::
                                         MetavariablesWithoutBoundaryConditions<
                                             2, domain::creators::Rectangle>>(
            "Rectangle:\n"
            "  LowerBound: [0,0]\n"
            "  UpperBound: [1,2]\n"
            "  IsPeriodicIn: [True,False]\n"
            "  InitialGridPoints: [3,4]\n"
            "  InitialRefinement: [2,3]\n"
            "  TimeDependence:\n"
            "    UniformTranslation:\n"
            "      InitialTime: 1.0\n"
            "      Velocity: [2.3, -0.3]\n");
    const auto* rectangle_creator =
        dynamic_cast<const creators::Rectangle*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3, -0.3}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        expected_neighbors,
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_eta()}, {Direction<2>::upper_eta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{2, 0.0}, velocity, {2, 0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation2D{f_of_t_name}));
    // with expiration times
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        expected_neighbors,
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_eta()}, {Direction<2>::upper_eta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{2, 0.0}, velocity, {2, 0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation2D{f_of_t_name}),
        false, initial_expiration_times);
  }
  {
    INFO("Rectangle factory time dependent, with boundary conditions");
    const auto domain_creator =
        TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<2>,
                                     TestHelpers::domain::BoundaryConditions::
                                         MetavariablesWithBoundaryConditions<
                                             2, domain::creators::Rectangle>>(
            "Rectangle:\n"
            "  LowerBound: [0,0]\n"
            "  UpperBound: [1,2]\n"
            "  InitialGridPoints: [3,4]\n"
            "  InitialRefinement: [2,3]\n"
            "  TimeDependence:\n"
            "    UniformTranslation:\n"
            "      InitialTime: 1.0\n"
            "      Velocity: [2.3, -0.3]\n"
            "  BoundaryCondition:\n"
            "    TestBoundaryCondition:\n"
            "      Direction: lower-xi\n"
            "      BlockId: 0\n");
    const auto* rectangle_creator =
        dynamic_cast<const creators::Rectangle*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3, -0.3}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        {{}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_xi(), Direction<2>::upper_xi(),
              Direction<2>::lower_eta(), Direction<2>::upper_eta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{2, 0.0}, velocity, {2, 0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation2D{f_of_t_name}),
        true);
    // with expiration times
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        {{}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_xi(), Direction<2>::upper_xi(),
              Direction<2>::lower_eta(), Direction<2>::upper_eta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{2, 0.0}, velocity, {2, 0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation2D{f_of_t_name}),
        true, initial_expiration_times);
  }
}  // namespace domain
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Rectangle", "[Domain][Unit]") {
  test_rectangle();
  test_rectangle_factory();
}
}  // namespace domain
