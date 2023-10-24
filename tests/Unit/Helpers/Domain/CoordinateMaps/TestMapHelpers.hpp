// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Helper functions for testing coordinate maps

#pragma once

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependentHelpers.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map and a CoordinateMapBase, checks that the maps are equal by
 * downcasting `map_base` and then comparing to `map`. Returns false if the
 * downcast fails.
 */
template <typename Map, typename SourceFrame, typename TargetFrame>
bool are_maps_equal(const Map& map,
                    const domain::CoordinateMapBase<SourceFrame, TargetFrame,
                                                    Map::dim>& map_base) {
  const auto* map_derived = dynamic_cast<const Map*>(&map_base);
  return map_derived == nullptr ? false : (*map_derived == map);
}

/// \ingroup TestingFrameworkGroup
/// \brief Given two coordinate maps (but not their types), check that the maps
/// are equal by evaluating them at a random set of points.
template <typename SourceFrame, typename TargetFrame, size_t VolumeDim>
void check_if_maps_are_equal(
    const domain::CoordinateMapBase<SourceFrame, TargetFrame, VolumeDim>&
        map_one,
    const domain::CoordinateMapBase<SourceFrame, TargetFrame, VolumeDim>&
        map_two,
    const double time = std::numeric_limits<double>::quiet_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {},
    Approx custom_approx = approx) {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);

  for (size_t n = 0; n < 10; ++n) {
    tnsr::I<double, VolumeDim, SourceFrame> source_point{};
    for (size_t d = 0; d < VolumeDim; ++d) {
      source_point.get(d) = real_dis(gen);
    }
    CAPTURE(source_point);
    CHECK_ITERABLE_CUSTOM_APPROX(map_one(source_point, time, functions_of_time),
                                 map_two(source_point, time, functions_of_time),
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        map_one.jacobian(source_point, time, functions_of_time),
        map_two.jacobian(source_point, time, functions_of_time), custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        map_one.inv_jacobian(source_point, time, functions_of_time),
        map_two.inv_jacobian(source_point, time, functions_of_time),
        custom_approx);
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Given a coordinate map, check that this map is equal to the identity
/// by evaluating the map at a random set of points.
template <typename Map>
void check_if_map_is_identity(const Map& map) {
  using IdentityMap = domain::CoordinateMaps::Identity<Map::dim>;
  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::Inertial, Frame::Grid>(IdentityMap{}),
      domain::make_coordinate_map<Frame::Inertial, Frame::Grid>(map));
  CHECK(map.is_identity());
}

/// @{
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, checks that the jacobian gives expected results
 * when compared to the numerical derivative in each direction.
 */
template <typename Map>
void test_jacobian(const Map& map,
                   const std::array<double, Map::dim>& test_point) {
  INFO("Test Jacobian");
  CAPTURE(test_point);
  // Our default approx value is too stringent for this test
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  const double dx = 1e-4;
  const auto jacobian = map.jacobian(test_point);
  for (size_t i = 0; i < Map::dim; ++i) {
    const auto numerical_deriv_i = numerical_derivative(map, test_point, i, dx);
    for (size_t j = 0; j < Map::dim; ++j) {
      INFO("i: " << i << " j: " << j);
      CHECK(jacobian.get(j, i) == local_approx(gsl::at(numerical_deriv_i, j)));
    }
  }
}

template <typename Map>
void test_jacobian(
    const Map& map, const std::array<double, Map::dim>& test_point,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  INFO("Test time-dependent Jacobian");
  CAPTURE(test_point);
  CAPTURE(time);
  const auto compute_map_point =
      [&map, time,
       &functions_of_time](const std::array<double, Map::dim>& point) {
        return map(point, time, functions_of_time);
      };
  // Our default approx value is too stringent for this test
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  const double dx = 1e-4;
  const auto jacobian = map.jacobian(test_point, time, functions_of_time);
  for (size_t i = 0; i < Map::dim; ++i) {
    const auto numerical_deriv_i =
        numerical_derivative(compute_map_point, test_point, i, dx);
    for (size_t j = 0; j < Map::dim; ++j) {
      INFO("i: " << i << " j: " << j);
      CHECK(jacobian.get(j, i) == local_approx(gsl::at(numerical_deriv_i, j)));
    }
  }
}

template <typename Map>
void test_jacobian(const Map& map,
                   const std::array<DataVector, Map::dim>& test_point) {
  INFO("Test Jacobian");
  CAPTURE(test_point);
  // Our default approx value is too stringent for this test
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  const double dx = 1e-4;
  const auto jacobian = map.jacobian(test_point);
  for (size_t i = 0; i < Map::dim; ++i) {
    for (size_t k = 0; k < gsl::at(test_point, 0).size(); k++) {
      const auto numerical_deriv_i =
          numerical_derivative(map, gsl::at(test_point, k), i, dx);
      for (size_t j = 0; j < Map::dim; ++j) {
        INFO("i: " << i << " j: " << j);
        CHECK(jacobian.get(j, i)[k] ==
              local_approx(gsl::at(numerical_deriv_i, j)));
      }
    }
  }
}

template <typename Map>
void test_jacobian(
    const Map& map, const std::array<DataVector, Map::dim>& test_point,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  INFO("Test time-dependent Jacobian");
  CAPTURE(test_point);
  CAPTURE(time);
  const auto compute_map_point =
      [&map, time,
       &functions_of_time](const std::array<double, Map::dim>& point) {
        return map(point, time, functions_of_time);
      };
  // Our default approx value is too stringent for this test
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  const double dx = 1e-4;
  const auto jacobian = map.jacobian(test_point, time, functions_of_time);
  std::array<std::array<double, Map::dim>, 5> dv_to_double_array{};
  std::array<double, Map::dim> dv_to_double{};
  for (size_t i = 0; i < gsl::at(test_point, 0).size(); ++i) {
    for (size_t k = 0; k < Map::dim; k++) {
      gsl::at(dv_to_double, k) = gsl::at(gsl::at(test_point, k), i);
    }
    gsl::at(dv_to_double_array, i) = dv_to_double;
  }
  for (size_t i = 0; i < Map::dim; ++i) {
    for (size_t k = 0; k < gsl::at(test_point, 0).size(); k++) {
      const auto numerical_deriv_i =
          numerical_derivative(compute_map_point, dv_to_double_array[k], i, dx);
      for (size_t j = 0; j < Map::dim; ++j) {
        INFO("i: " << i << " j: " << j);
        CHECK(jacobian.get(j, i)[k] ==
              local_approx(gsl::at(numerical_deriv_i, j)));
      }
    }
  }
}
/// @}

/// @{
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, checks that the inverse jacobian and jacobian
 * multiply together to produce the identity matrix
 */
template <typename Map>
void test_inv_jacobian(const Map& map,
                       const std::array<double, Map::dim>& test_point) {
  INFO("Test inverse Jacobian");
  CAPTURE(test_point);
  const auto jacobian = map.jacobian(test_point);
  const auto inv_jacobian = map.inv_jacobian(test_point);

  const auto expected_identity = [&jacobian, &inv_jacobian]() {
    std::array<std::array<double, Map::dim>, Map::dim> identity{};
    for (size_t i = 0; i < Map::dim; ++i) {
      for (size_t j = 0; j < Map::dim; ++j) {
        gsl::at(gsl::at(identity, i), j) = 0.;
        for (size_t k = 0; k < Map::dim; ++k) {
          gsl::at(gsl::at(identity, i), j) +=
              jacobian.get(i, k) * inv_jacobian.get(k, j);
        }
      }
    }
    return identity;
  }();

  for (size_t i = 0; i < Map::dim; ++i) {
    for (size_t j = 0; j < Map::dim; ++j) {
      CHECK(gsl::at(gsl::at(expected_identity, i), j) ==
            approx(i == j ? 1. : 0.));
    }
  }
}

template <typename Map>
void test_inv_jacobian(
    const Map& map, const std::array<double, Map::dim>& test_point,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  INFO("Test inverse time-dependent Jacobian");
  CAPTURE(test_point);
  CAPTURE(time);
  const auto jacobian = map.jacobian(test_point, time, functions_of_time);
  const auto inv_jacobian =
      map.inv_jacobian(test_point, time, functions_of_time);

  const auto expected_identity = [&jacobian, &inv_jacobian]() {
    std::array<std::array<double, Map::dim>, Map::dim> identity{};
    for (size_t i = 0; i < Map::dim; ++i) {
      for (size_t j = 0; j < Map::dim; ++j) {
        gsl::at(gsl::at(identity, i), j) = 0.;
        for (size_t k = 0; k < Map::dim; ++k) {
          gsl::at(gsl::at(identity, i), j) +=
              jacobian.get(i, k) * inv_jacobian.get(k, j);
        }
      }
    }
    return identity;
  }();

  for (size_t i = 0; i < Map::dim; ++i) {
    for (size_t j = 0; j < Map::dim; ++j) {
      CHECK(gsl::at(gsl::at(expected_identity, i), j) ==
            approx(i == j ? 1. : 0.));
    }
  }
}

template <typename Map>
void test_inv_jacobian(const Map& map,
                       const std::array<DataVector, Map::dim>& test_point) {
  INFO("Test inverse Jacobian");
  CAPTURE(test_point);
  const auto jacobian = map.jacobian(test_point);
  const auto inv_jacobian = map.inv_jacobian(test_point);

  const auto expected_identity = [&jacobian, &inv_jacobian]() {
    auto identity =
        make_with_value<tnsr::Ij<DataVector, Map::dim, Frame::NoFrame>>(
            jacobian.get(0, 0).size(), 0.0);
    for (size_t i = 0; i < Map::dim; ++i) {
      for (size_t j = 0; j < Map::dim; ++j) {
        for (size_t l = 0; l < jacobian.get(0, 0).size(); l++) {
          for (size_t k = 0; k < Map::dim; ++k) {
            identity.get(i, j)[k] += gsl::at(jacobian.get(i, k), l) *
                                     gsl::at(inv_jacobian.get(k, j), l);
          }
        }
      }
    }
    return identity;
  }();

  for (size_t i = 0; i < Map::dim; ++i) {
    for (size_t j = 0; j < Map::dim; ++j) {
      for (size_t k = 0; k < gsl::at(test_point, 0).size(); k++) {
        CHECK(gsl::at(expected_identity.get(i, j), k) ==
              approx(i == j ? 1. : 0.));
      }
    }
  }
}

template <typename Map>
void test_inv_jacobian(
    const Map& map, const std::array<DataVector, Map::dim>& test_point,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  INFO("Test inverse Jacobian");
  CAPTURE(test_point);
  CAPTURE(time);
  const auto jacobian = map.jacobian(test_point, time, functions_of_time);
  const auto inv_jacobian =
      map.inv_jacobian(test_point, time, functions_of_time);

  const auto expected_identity = [&jacobian, &inv_jacobian]() {
    auto identity =
        make_with_value<tnsr::Ij<DataVector, Map::dim, Frame::NoFrame>>(
            jacobian.get(0, 0).size(), 0.0);
    for (size_t i = 0; i < Map::dim; ++i) {
      for (size_t j = 0; j < Map::dim; ++j) {
        for (size_t l = 0; l < jacobian.get(0, 0).size(); l++) {
          for (size_t k = 0; k < Map::dim; ++k) {
            identity.get(i, j)[l] += gsl::at(jacobian.get(i, k), l) *
                                     gsl::at(inv_jacobian.get(k, j), l);
          }
        }
      }
    }
    return identity;
  }();
  for (size_t i = 0; i < Map::dim; ++i) {
    for (size_t j = 0; j < Map::dim; ++j) {
      for (size_t k = 0; k < gsl::at(test_point, 0).size(); k++) {
        CHECK(gsl::at(expected_identity.get(i, j), k) ==
              approx(i == j ? 1. : 0.));
      }
    }
  }
}
/// @}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, checks that the frame velocity matches a
 * sixth-order finite difference approximation.
 */
template <typename Map>
void test_frame_velocity(
    const Map& map, const std::array<double, Map::dim>& test_point,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  const auto compute_map_point = [&map, &test_point, &functions_of_time](
                                     const std::array<double, 1>& time_point) {
    return map(test_point, time_point[0], functions_of_time);
  };
  // Our default approx value is too stringent for this test
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  const double dt = 1e-4;

  const auto frame_velocity =
      map.frame_velocity(test_point, time, functions_of_time);
  const auto numerical_frame_velocity = numerical_derivative(
      compute_map_point, std::array<double, 1>{{time}}, 0, dt);
  CHECK_ITERABLE_CUSTOM_APPROX(frame_velocity, numerical_frame_velocity,
                               local_approx);
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Checks that the CoordinateMap `map` functions as expected when used as
 * the template parameter to the `CoordinateMap` type.
 */
template <typename Map, typename... Args>
void test_coordinate_map_implementation(const Map& map) {
  const auto coord_map =
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);

  const auto test_point = [&gen, &real_dis] {
    std::array<double, Map::dim> p{};
    for (size_t i = 0; i < Map::dim; ++i) {
      gsl::at(p, i) = real_dis(gen);
    }
    return p;
  }();

  const auto test_point_tensor = [&test_point]() {
    tnsr::I<double, Map::dim, Frame::BlockLogical> point_as_tensor{};
    for (size_t i = 0; i < Map::dim; ++i) {
      point_as_tensor.get(i) = gsl::at(test_point, i);
    }
    return point_as_tensor;
  }();

  for (size_t i = 0; i < Map::dim; ++i) {
    CHECK(coord_map(test_point_tensor).get(i) ==
          approx(gsl::at(map(test_point), i)));
    for (size_t j = 0; j < Map::dim; ++j) {
      CHECK(coord_map.jacobian(test_point_tensor).get(i, j) ==
            map.jacobian(test_point).get(i, j));
      CHECK(coord_map.inv_jacobian(test_point_tensor).get(i, j) ==
            map.inv_jacobian(test_point).get(i, j));
    }
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Checks that the CoordinateMap `map` functions as expected when used
 * with different argument types.
 */
template <typename Map, typename... Args>
void test_coordinate_map_argument_types(
    const Map& map, const std::array<double, Map::dim>& test_point,
    const Args&... args) {
  INFO("Test coordinate map argument types");
  CAPTURE(test_point);
  const auto make_array_data_vector = [](const auto& double_array) {
    std::array<DataVector, Map::dim> result;
    std::transform(double_array.begin(), double_array.end(), result.begin(),
                   [](const double x) {
                     return DataVector{x, x};
                   });
    return result;
  };
  const auto add_reference_wrapper = [](const auto& unwrapped_array) {
    using Arg = std::decay_t<decltype(unwrapped_array)>;
    return make_array<std::reference_wrapper<const typename Arg::value_type>,
                      Map::dim>(unwrapped_array);
  };

  {
    INFO("Test call-operator");
    const auto mapped_point = map(test_point, args...);
    CHECK_ITERABLE_APPROX(map(add_reference_wrapper(test_point), args...),
                          mapped_point);
    CHECK_ITERABLE_APPROX(map(make_array_data_vector(test_point), args...),
                          make_array_data_vector(mapped_point));
    CHECK_ITERABLE_APPROX(
        map(add_reference_wrapper(make_array_data_vector(test_point)), args...),
        make_array_data_vector(mapped_point));
  }

  // Here, time_args is a const auto& not const Args& because time_args
  // is allowed to be different than Args (which was the reason for the
  // overloader below that calls this function).
  const auto check_jac = [](const auto& make_arr_data_vec,
                            const auto& add_ref_wrap, const Map& the_map,
                            const std::array<double, Map::dim>& point,
                            const auto&... time_args) {
    const auto make_tensor_data_vector = [](const auto& double_tensor) {
      using Arg = std::decay_t<decltype(double_tensor)>;
      Tensor<DataVector, typename Arg::symmetry, typename Arg::index_list>
          result;
      std::transform(double_tensor.begin(), double_tensor.end(), result.begin(),
                     [](const double x) {
                       return DataVector{x, x};
                     });
      return result;
    };

    {
      INFO("Test Jacobian");
      const auto expected = the_map.jacobian(point, time_args...);
      CHECK_ITERABLE_APPROX(the_map.jacobian(add_ref_wrap(point), time_args...),
                            expected);
      CHECK_ITERABLE_APPROX(
          the_map.jacobian(make_arr_data_vec(point), time_args...),
          make_tensor_data_vector(expected));
      CHECK_ITERABLE_APPROX(
          the_map.jacobian(add_ref_wrap(make_arr_data_vec(point)),
                           time_args...),
          make_tensor_data_vector(expected));
    }
    {
      INFO("Test inverse Jacobian");
      Approx custom_approx = Approx::custom().epsilon(1.e-11);
      const auto expected = the_map.inv_jacobian(point, time_args...);
      CHECK_ITERABLE_APPROX(
          the_map.inv_jacobian(add_ref_wrap(point), time_args...), expected);
      CHECK_ITERABLE_CUSTOM_APPROX(
          the_map.inv_jacobian(make_arr_data_vec(point), time_args...),
          make_tensor_data_vector(expected), custom_approx);
      CHECK_ITERABLE_CUSTOM_APPROX(
          the_map.inv_jacobian(add_ref_wrap(make_arr_data_vec(point)),
                               time_args...),
          make_tensor_data_vector(expected), custom_approx);
    }

    return nullptr;
  };

  if constexpr (domain::is_jacobian_time_dependent_t<decltype(map), double>{}) {
    check_jac(make_array_data_vector, add_reference_wrapper, map, test_point,
              args...);
  } else {
    check_jac(make_array_data_vector, add_reference_wrapper, map, test_point);
  }
}

/// @{
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, checks that the inverse map gives expected results
 */
template <typename Map, typename T>
void test_inverse_map(const Map& map,
                      const std::array<T, Map::dim>& test_point) {
  INFO("Test inverse map");
  CAPTURE(test_point);
  const auto mapped_point = map(test_point);
  CAPTURE(mapped_point);
  const auto expected_test_point = map.inverse(mapped_point);
  REQUIRE(expected_test_point.has_value());
  CHECK_ITERABLE_APPROX(test_point, expected_test_point.value());
}

template <typename Map, typename T>
void test_inverse_map(
    const Map& map, const std::array<T, Map::dim>& test_point,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  INFO("Test inverse map time dependent");
  CAPTURE(test_point);
  const auto expected_test_point = map.inverse(
      map(test_point, time, functions_of_time), time, functions_of_time);
  REQUIRE(expected_test_point.has_value());
  CHECK_ITERABLE_APPROX(test_point, expected_test_point.value());
}
/// @}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, tests the map functions, including map inverse,
 * jacobian, and inverse jacobian, for a series of points.
 * These points are chosen in a dim-dimensonal cube of side 2 centered at
 * the origin.  The map is expected to be valid on the boundaries of the cube.
 */
template <typename Map>
void test_suite_for_map_on_unit_cube(const Map& map) {
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1.0, 1.0);

  std::array<double, Map::dim> origin{};
  std::array<double, Map::dim> random_point{};
  for (size_t i = 0; i < Map::dim; i++) {
    gsl::at(origin, i) = 0.0;
    gsl::at(random_point, i) = real_dis(gen);
  }

  const auto test_helper = [&origin, &random_point](const auto& map_to_test) {
    test_serialization(map_to_test);
    CHECK_FALSE(map_to_test != map_to_test);
    test_coordinate_map_argument_types(map_to_test, origin);

    test_jacobian(map_to_test, origin);
    test_inv_jacobian(map_to_test, origin);
    test_inverse_map(map_to_test, origin);

    for (VolumeCornerIterator<Map::dim> vci{}; vci; ++vci) {
      test_jacobian(map_to_test, vci.coords_of_corner());
      test_inv_jacobian(map_to_test, vci.coords_of_corner());
      test_inverse_map(map_to_test, vci.coords_of_corner());
    }

    test_jacobian(map_to_test, random_point);
    test_inv_jacobian(map_to_test, random_point);
    test_inverse_map(map_to_test, random_point);
  };
  test_helper(map);
  const auto map2 = serialize_and_deserialize(map);
  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map2));
  test_helper(map2);
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, tests the map functions, including map inverse,
 * jacobian, and inverse jacobian, for a series of points.
 * These points are chosen in a sphere of radius `radius_of_sphere`, and the
 * map is expected to be valid on the boundary of that sphere as well as
 * in its interior.  The flag `include_origin` indicates whether to test the
 * map at the origin.
 * This test works only in 3 dimensions.
 */
template <typename Map>
void test_suite_for_map_on_sphere(const Map& map,
                                  const bool include_origin = true,
                                  const double radius_of_sphere = 1.0) {
  static_assert(Map::dim == 3,
                "test_suite_for_map_on_sphere works only for a 3d map");

  // Set up random number generator
  MAKE_GENERATOR(gen);

  // If we don't include the origin, we want to use some finite inner
  // boundary so that random points stay away from the origin.
  // test_jacobian has a dx of 1.e-4 for finite-differencing, so here
  // we pick a value larger than that.
  const double inner_bdry = include_origin ? 0.0 : 5.e-3;

  std::uniform_real_distribution<> radius_dis(inner_bdry, radius_of_sphere);
  std::uniform_real_distribution<> theta_dis(0, M_PI);
  std::uniform_real_distribution<> phi_dis(0, 2.0 * M_PI);

  const double theta = theta_dis(gen);
  CAPTURE(theta);
  const double phi = phi_dis(gen);
  CAPTURE(phi);
  const double radius = radius_dis(gen);
  CAPTURE(radius);

  const std::array<std::array<double, 3>, 3> points_to_test{
      {// A random point in the interior
       {radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi),
        radius * cos(theta)},
       // A random point on the outer boundary
       {radius_of_sphere * sin(theta) * cos(phi),
        radius_of_sphere * sin(theta) * sin(phi),
        radius_of_sphere * cos(theta)},
       // Either a point at the origin or (if include_origin is false)
       // a random point on the inner boundary.
       {inner_bdry * sin(theta) * cos(phi), inner_bdry * sin(theta) * sin(phi),
        inner_bdry * cos(theta)}}};

  const auto test_helper = [&points_to_test](const auto& map_to_test) {
    test_serialization(map_to_test);
    CHECK_FALSE(map_to_test != map_to_test);
    for (const auto& point : points_to_test) {
      test_coordinate_map_argument_types(map_to_test, point);
      test_jacobian(map_to_test, point);
      test_inv_jacobian(map_to_test, point);
      test_inverse_map(map_to_test, point);
    };
  };

  test_helper(map);
  const auto map2 = serialize_and_deserialize(map);
  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map2));
  test_helper(map2);
}

  /*!
   * \ingroup TestingFrameworkGroup
   * \brief Given a Map `map`, tests the map functions, including map inverse,
   * jacobian, and inverse jacobian, for a series of points.
   * These points are chosen in a right cylinder with cylindrical radius 1
   * and z-axis extending from -1 to +1. The
   * map is expected to be valid on the boundary of that cylinder as well as
   * in its interior.
   * This test works only in 3 dimensions.
   */
template <typename Map>
void test_suite_for_map_on_cylinder(
    const Map& map, const double inner_radius, const double outer_radius,
    const bool test_random_z_bdry_roundoff = false,
    const bool test_random_rho_bdry_roundoff = false) {
  static_assert(Map::dim == 3,
                "test_suite_for_map_on_cylinder works only for a 3d map");

  // Set up random number generator
  MAKE_GENERATOR(gen);

  std::uniform_real_distribution<> radius_dis(inner_radius, outer_radius);
  std::uniform_real_distribution<> phi_dis(0.0, 2.0 * M_PI);
  std::uniform_real_distribution<> height_dis(-1.0, 1.0);

  const double height = height_dis(gen);
  CAPTURE(height);
  const double phi = phi_dis(gen);
  CAPTURE(phi);
  const double radius = radius_dis(gen);
  CAPTURE(radius);

  const std::array<double, 3> random_point{
      {radius * cos(phi), radius * sin(phi), height}};

  const std::array<double, 3> random_bdry_point_rho{
      {outer_radius * cos(phi), outer_radius * sin(phi), height}};

  const std::array<double, 3> random_bdry_point_z{
      {radius * cos(phi), radius * sin(phi), height > 0.5 ? 1.0 : -1.0}};

  const std::array<double, 3> random_bdry_point_corner{
      {outer_radius * cos(phi), outer_radius * sin(phi),
       height > 0.5 ? 1.0 : -1.0}};

  // If inner_radius is zero, this point is on the axis.
  const std::array<double, 3> random_inner_bdry_point_or_origin{
      {inner_radius * cos(phi), inner_radius * sin(phi), height}};

  // If inner_radius is zero, this point is on the axis.
  const std::array<double, 3> random_inner_bdry_point_corner{
      {inner_radius * cos(phi), inner_radius * sin(phi),
       height > 0.5 ? 1.0 : -1.0}};

  const auto test_helper =
      [](const auto& map_to_test,
         const std::vector<std::array<double, 3>>& points_to_test) {
        test_serialization(map_to_test);
        CHECK_FALSE(map_to_test != map_to_test);

        for (const auto& point : points_to_test) {
          test_coordinate_map_argument_types(map_to_test, point);
          test_jacobian(map_to_test, point);
          test_inv_jacobian(map_to_test, point);
          test_inverse_map(map_to_test, point);
        }
      };

  const auto test_helper_all_points = [&test_helper, &random_bdry_point_rho,
                                       &random_bdry_point_z,
                                       &random_bdry_point_corner,
                                       &random_inner_bdry_point_or_origin,
                                       &random_inner_bdry_point_corner,
                                       &random_point](const auto& map_to_test) {
    test_helper(map_to_test,
                {random_bdry_point_rho, random_bdry_point_z,
                 random_bdry_point_corner, random_inner_bdry_point_or_origin,
                 random_inner_bdry_point_corner, random_point});
  };

  // Test points that are within roundoff of the +/- z faces of the
  // cylinder.  We test multiple points to increase the probability
  // that we hit all of the relevant cases, since it is hard to
  // predict the details of whether roundoff makes certain
  // expressions in the map slightly smaller or slightly larger than
  // they should be.
  const auto test_helper_z_roundoff_points =
      [&gen, &height_dis, &radius, &phi,
       &test_helper](const auto& map_to_test) {
        for (size_t i = 0; i < 50; ++i) {
          const double z_roundoff = 1.e-15 * height_dis(gen);
          CAPTURE(z_roundoff);
          const std::array<double, 3> random_bdry_point_roundoff{
              {radius * cos(phi), radius * sin(phi), 1.0 + z_roundoff}};
          const std::array<double, 3> random_bdry_point_minus_roundoff{
              {radius * cos(phi), radius * sin(phi), -1.0 + z_roundoff}};
          test_helper(map_to_test, {random_bdry_point_roundoff,
                                    random_bdry_point_minus_roundoff});
        }
      };
  const auto test_helper_rho_roundoff_points =
      [&gen, &radius_dis, &inner_radius, &outer_radius, &phi, &height,
       &test_helper](const auto& map_to_test) {
        for (size_t i = 0; i < 25; ++i) {
          const double rho_roundoff = 1.e-15 * radius_dis(gen) / outer_radius;
          CAPTURE(rho_roundoff);
          const std::array<double, 3> random_bdry_point_inner_roundoff{
              {(inner_radius + rho_roundoff) * cos(phi),
               (inner_radius + rho_roundoff) * sin(phi), height}};
          const std::array<double, 3> random_bdry_point_outer_roundoff{
              {(outer_radius + rho_roundoff) * cos(phi),
               (outer_radius + rho_roundoff) * sin(phi), height}};
          const std::array<double, 3> random_bdry_point_outer_minus_roundoff{
              {(outer_radius - rho_roundoff) * cos(phi),
               (outer_radius - rho_roundoff) * sin(phi), height}};
          if(inner_radius != 0.0) {
            const std::array<double, 3> random_bdry_point_inner_minus_roundoff{
                {(inner_radius - rho_roundoff) * cos(phi),
                 (inner_radius - rho_roundoff) * sin(phi), height}};
            test_helper(map_to_test, {random_bdry_point_inner_roundoff,
                                      random_bdry_point_outer_roundoff,
                                      random_bdry_point_outer_minus_roundoff,
                                      random_bdry_point_inner_minus_roundoff});
          } else {
            // If inner_radius is zero, ignore
            // random_bdry_point_inner_minus_roundoff because that
            // would make the radius negative.
            test_helper(map_to_test, {random_bdry_point_inner_roundoff,
                                      random_bdry_point_outer_roundoff,
                                      random_bdry_point_outer_minus_roundoff});
          }
        }
      };

  test_helper_all_points(map);
  if (test_random_z_bdry_roundoff) {
    test_helper_z_roundoff_points(map);
  }
  if (test_random_rho_bdry_roundoff) {
    test_helper_rho_roundoff_points(map);
  }
  const auto map2 = serialize_and_deserialize(map);
  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map2));
  test_helper_all_points(map2);
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief An iterator for looping through all possible orientations
 * of the n-dim cube.
 */
template <size_t VolumeDim>
class OrientationMapIterator {
 public:
  OrientationMapIterator() {
    std::iota(dims_.begin(), dims_.end(), 0);
    set_map();
  }
  void operator++() {
    ++vci_;
    if (not vci_) {
      not_at_end_ = std::next_permutation(dims_.begin(), dims_.end());
      vci_ = VolumeCornerIterator<VolumeDim>{};
    }
    set_map();
  }
  explicit operator bool() const { return not_at_end_; }
  const OrientationMap<VolumeDim>& operator()() const { return map_; }
  const OrientationMap<VolumeDim>& operator*() const { return map_; }
  void set_map() {
    for (size_t i = 0; i < VolumeDim; i++) {
      gsl::at(directions_, i) =
          Direction<VolumeDim>{gsl::at(dims_, i), gsl::at(vci_(), i)};
    }
    map_ = OrientationMap<VolumeDim>{directions_};
  }

 private:
  bool not_at_end_ = true;
  std::array<size_t, VolumeDim> dims_{};
  std::array<Direction<VolumeDim>, VolumeDim> directions_{};
  VolumeCornerIterator<VolumeDim> vci_{};
  OrientationMap<VolumeDim> map_ = OrientationMap<VolumeDim>{};
};

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Wedge OrientationMap in each of the six directions used in the
 * Sphere domain creator.
 */
inline std::array<OrientationMap<3>, 6> all_wedge_directions() {
  const OrientationMap<3> upper_zeta_rotation{};
  const OrientationMap<3> lower_zeta_rotation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
       Direction<3>::lower_zeta()}});
  const OrientationMap<3> upper_eta_rotation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_zeta(),
       Direction<3>::upper_xi()}});
  const OrientationMap<3> lower_eta_rotation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
       Direction<3>::lower_xi()}});
  const OrientationMap<3> upper_xi_rotation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
       Direction<3>::upper_eta()}});
  const OrientationMap<3> lower_xi_rotation(std::array<Direction<3>, 3>{
      {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
       Direction<3>::upper_eta()}});
  return {{upper_zeta_rotation, lower_zeta_rotation, upper_eta_rotation,
           lower_eta_rotation, upper_xi_rotation, lower_xi_rotation}};
}
