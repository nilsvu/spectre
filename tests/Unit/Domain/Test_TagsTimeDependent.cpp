// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void test_tags() {
  TestHelpers::db::test_simple_tag<
      domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
      "CoordinatesMeshVelocityAndJacobians");
  TestHelpers::db::test_compute_tag<
      domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
          domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>>>(
      "CoordinatesMeshVelocityAndJacobians");
  TestHelpers::db::test_compute_tag<
      domain::Tags::InertialFromGridCoordinatesCompute<Dim>>(
      "InertialCoordinates");
  TestHelpers::db::test_compute_tag<
      domain::Tags::ElementToInertialInverseJacobian<Dim>>(
      "InverseJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_compute_tag<
      domain::Tags::GridToInertialInverseJacobian<Dim>>(
      "InverseJacobian(Grid,Inertial)");
  TestHelpers::db::test_simple_tag<domain::Tags::MeshVelocity<Dim>>(
      "MeshVelocity");
  TestHelpers::db::test_compute_tag<
      domain::Tags::InertialMeshVelocityCompute<Dim>>("MeshVelocity");
  TestHelpers::db::test_simple_tag<domain::Tags::DivMeshVelocity>(
      "div(MeshVelocity)");
}

using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation<1>;
using TranslationMap2d = domain::CoordinateMaps::TimeDependent::Translation<2>;
using TranslationMap3d = domain::CoordinateMaps::TimeDependent::Translation<3>;
using AffineMap = domain::CoordinateMaps::Affine;
using AffineMap2d =
    domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
using AffineMap3d =
    domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;

template <size_t MeshDim>
using ConcreteMap = tmpl::conditional_t<
    MeshDim == 1,
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap,
                          AffineMap>,
    tmpl::conditional_t<MeshDim == 2,
                        domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                              TranslationMap2d, AffineMap2d>,
                        domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                              TranslationMap3d, AffineMap3d>>>;

template <size_t MeshDim>
ConcreteMap<MeshDim> create_coord_map(const std::string& f_of_t_name);

template <>
ConcreteMap<1> create_coord_map<1>(const std::string& f_of_t_name) {
  return ConcreteMap<1>{TranslationMap{f_of_t_name},
                        AffineMap{-1.0, 1.0, 2.0, 7.2}};
}

template <>
ConcreteMap<2> create_coord_map<2>(const std::string& f_of_t_name) {
  return ConcreteMap<2>{
      {TranslationMap2d{f_of_t_name}},
      {AffineMap{-1.0, 1.0, -2.0, 2.2}, AffineMap{-1.0, 1.0, 2.0, 7.2}}};
}

template <>
ConcreteMap<3> create_coord_map<3>(const std::string& f_of_t_name) {
  return ConcreteMap<3>{
      {TranslationMap3d{f_of_t_name}},
      {AffineMap{-1.0, 1.0, -2.0, 2.2}, AffineMap{-1.0, 1.0, 2.0, 7.2},
       AffineMap{-1.0, 1.0, 1.0, 3.5}}};
}

template <size_t Dim, bool IsTimeDependent>
void test() {
  using simple_tags = db::AddSimpleTags<
      Tags::Time, domain::Tags::Coordinates<Dim, Frame::Grid>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Grid>,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>>;
  using compute_tags = db::AddComputeTags<
      domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
          domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>>,
      domain::Tags::InertialFromGridCoordinatesCompute<Dim>,
      domain::Tags::ElementToInertialInverseJacobian<Dim>,
      domain::Tags::GridToInertialInverseJacobian<Dim>,
      domain::Tags::InertialMeshVelocityCompute<Dim>>;

  MAKE_GENERATOR(gen);
  const DataVector velocity{Dim, 1.2};
  const double initial_time = 0.0;
  const double expiration_time = 5.0;
  const std::string function_of_time_name = "Translation";

  UniformCustomDistribution<double> dist(-10.0, 10.0);

  const size_t num_pts = Dim * 5;

  tnsr::I<DataVector, Dim, Frame::Grid> grid_coords{num_pts};
  fill_with_random_values(make_not_null(&grid_coords), make_not_null(&gen),
                          make_not_null(&dist));
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>
      element_to_grid_inverse_jacobian{num_pts};
  fill_with_random_values(make_not_null(&element_to_grid_inverse_jacobian),
                          make_not_null(&gen), make_not_null(&dist));

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[function_of_time_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time,
          std::array<DataVector, 3>{{{Dim, 0.0}, velocity, {Dim, 0.0}}},
          expiration_time);

  using MapPtr = std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>;
  const MapPtr grid_to_inertial_map =
      IsTimeDependent ? MapPtr{std::make_unique<ConcreteMap<Dim>>(
                            create_coord_map<Dim>(function_of_time_name))}
                      : MapPtr{std::make_unique<domain::CoordinateMap<
                            Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<Dim>>>()};

  const double time = 3.0;
  auto box = db::create<simple_tags, compute_tags>(
      time, grid_coords, element_to_grid_inverse_jacobian,
      std::move(functions_of_time), grid_to_inertial_map->get_clone());

  const auto check_helper = [&box, &element_to_grid_inverse_jacobian,
                             &grid_coords, &grid_to_inertial_map,
                             num_pts](const double expected_time) {
    if (IsTimeDependent) {
      const tnsr::I<DataVector, Dim, Frame::Inertial> expected_coords =
          (*grid_to_inertial_map)(grid_coords, expected_time,
                                  db::get<domain::Tags::FunctionsOfTime>(box));

      const InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>
          expected_inv_jacobian_grid_to_inertial =
              grid_to_inertial_map->inv_jacobian(
                  grid_coords, expected_time,
                  db::get<domain::Tags::FunctionsOfTime>(box));

      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
          expected_inv_jacobian{num_pts};

      for (size_t logical_i = 0; logical_i < Dim; ++logical_i) {
        for (size_t inertial_i = 0; inertial_i < Dim; ++inertial_i) {
          expected_inv_jacobian.get(logical_i, inertial_i) =
              element_to_grid_inverse_jacobian.get(logical_i, 0) *
              expected_inv_jacobian_grid_to_inertial.get(0, inertial_i);
          for (size_t grid_i = 1; grid_i < Dim; ++grid_i) {
            expected_inv_jacobian.get(logical_i, inertial_i) +=
                element_to_grid_inverse_jacobian.get(logical_i, grid_i) *
                expected_inv_jacobian_grid_to_inertial.get(grid_i, inertial_i);
          }
        }
      }

      REQUIRE(
          db::get<domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(box)
              .has_value());

      for (size_t i = 0; i < Dim; ++i) {
        // Check that the `const_cast`s and set_data_ref inside the compute
        // tag functions worked correctly
        CHECK(db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box)
                  .get(i)
                  .data() ==
              std::get<0>(
                  *db::get<
                      domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
                      box))
                  .get(i)
                  .data());
      }
      CHECK_ITERABLE_APPROX(
          (db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box)),
          expected_coords);

      for (size_t i = 0;
           i < db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                     Frame::Inertial>>(box)
                   .size();
           ++i) {
        CHECK(
            db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                  Frame::Inertial>>(box)[i]
                .data() !=
            std::get<1>(*db::get<
                        domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
                box))[i]
                .data());
      }
      CHECK_ITERABLE_APPROX(
          (db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                 Frame::Inertial>>(box)),
          expected_inv_jacobian);

      const auto expected_coords_mesh_velocity_jacobians =
          grid_to_inertial_map->coords_frame_velocity_jacobians(
              db::get<domain::Tags::Coordinates<Dim, Frame::Grid>>(box),
              db::get<::Tags::Time>(box),
              db::get<domain::Tags::FunctionsOfTime>(box));

      for (size_t i = 0; i < Dim; ++i) {
        // Check that the `const_cast`s and set_data_ref inside the compute
        // tag functions worked correctly
        CHECK(db::get<domain::Tags::MeshVelocity<Dim>>(box)->get(i).data() ==
              std::get<3>(
                  *db::get<
                      domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
                      box))
                  .get(i)
                  .data());
      }
      CHECK_ITERABLE_APPROX(
          db::get<domain::Tags::MeshVelocity<Dim>>(box).value(),
          std::get<3>(expected_coords_mesh_velocity_jacobians));
      for (size_t i = 0;
           i < db::get<domain::Tags::InverseJacobian<Dim, Frame::Grid,
                                                     Frame::Inertial>>(box)
                   .size();
           ++i) {
        // Check that the `const_cast`s and set_data_ref inside the
        // compute tag functions worked correctly
        CHECK(
            db::get<domain::Tags::InverseJacobian<Dim, Frame::Grid,
                                                  Frame::Inertial>>(box)[i]
                .data() ==
            std::get<1>(*db::get<
                        domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
                box))[i]
                .data());
      }
      CHECK_ITERABLE_APPROX(
          (db::get<
              domain::Tags::InverseJacobian<Dim, Frame::Grid, Frame::Inertial>>(
              box)),
          std::get<1>(expected_coords_mesh_velocity_jacobians));
    } else {
      tnsr::I<DataVector, Dim, Frame::Inertial> expected_coords{num_pts};
      for (size_t i = 0; i < Dim; ++i) {
        expected_coords[i] = grid_coords[i];
      }

      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
          expected_inv_jacobian{num_pts};
      // The Grid->Inertial Jacobian is currently just the identity
      for (size_t i = 0; i < expected_inv_jacobian.size(); ++i) {
        expected_inv_jacobian[i] = element_to_grid_inverse_jacobian[i];
      }

      for (size_t i = 0; i < Dim; ++i) {
        // Check that the `const_cast`s and set_data_ref inside the
        // compute tag functions worked correctly
        CHECK(db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box)
                  .get(i)
                  .data() ==
              db::get<domain::Tags::Coordinates<Dim, Frame::Grid>>(box)
                  .get(i)
                  .data());
      }
      CHECK_ITERABLE_APPROX(
          (db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box)),
          expected_coords);

      for (size_t i = 0;
           i < db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                     Frame::Inertial>>(box)
                   .size();
           ++i) {
        // Check that the `const_cast`s and set_data_ref inside the
        // compute tag functions worked correctly
        CHECK(db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                    Frame::Inertial>>(box)[i]
                  .data() ==
              db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                    Frame::Grid>>(box)[i]
                  .data());
      }
      CHECK_ITERABLE_APPROX(
          (db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                 Frame::Inertial>>(box)),
          expected_inv_jacobian);

      CHECK_FALSE(db::get<domain::Tags::MeshVelocity<Dim>>(box));
      CHECK_THROWS_WITH(
          (db::get<
              domain::Tags::InverseJacobian<Dim, Frame::Grid, Frame::Inertial>>(
              box)),
          Catch::Matchers::ContainsSubstring(
              "Should not request Grid to Inertial jacobian for a "
              "non-moving mesh "
              "because it is the identity."));
    }
  };
  check_helper(3.0);

  db::mutate<Tags::Time>(
      [](const gsl::not_null<double*> local_time) { *local_time = 4.5; },
      make_not_null(&box));
  check_helper(4.5);
}

SPECTRE_TEST_CASE("Unit.Domain.TagsTimeDependent", "[Unit][Actions]") {
  test_tags<1>();
  test_tags<2>();
  test_tags<3>();

  test<1, true>();
  test<2, true>();
  test<3, true>();

  test<1, false>();
  test<2, false>();
  test<3, false>();
}
}  // namespace
