// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/CartesianToSpherical.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"
#include "ParallelAlgorithms/RayTracer/RayTracer.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"

namespace ray_tracing {
namespace {

void test_partition_ranks() {
  INFO("Partition rays across ranks");
  for (size_t num_ranks = 1; num_ranks <= 8; ++num_ranks) {
    for (size_t total_num_rays = 0; total_num_rays <= 20; ++total_num_rays) {
      CAPTURE(num_ranks);
      CAPTURE(total_num_rays);
      // Ranks must cover all rays exactly once, in order, and be balanced
      size_t expected_start = 0;
      for (size_t rank = 0; rank < num_ranks; ++rank) {
        CAPTURE(rank);
        const auto [ray_index_start, num_rays_this_rank] =
            partition_ranks(total_num_rays, rank, num_ranks);
        CHECK(ray_index_start == expected_start);
        CHECK(num_rays_this_rank >= total_num_rays / num_ranks);
        CHECK(num_rays_this_rank <= total_num_rays / num_ranks + 1);
        expected_start += num_rays_this_rank;
      }
      CHECK(expected_start == total_num_rays);
    }
  }
}

double radius(const RayResults& ray_results, const size_t i) {
  return sqrt(square(ray_results(i, 0)) + square(ray_results(i, 1)) +
              square(ray_results(i, 2)));
}

void test_ray_status_output() {
  CHECK(get_output(RayStatus::TimeLimitReached) == "TimeLimitReached");
  CHECK(get_output(RayStatus::Escaped) == "Escaped");
  CHECK(get_output(RayStatus::Captured) == "Captured");
  CHECK(get_output(RayStatus::OutsideBackground) == "OutsideBackground");
  CHECK(get_output(RayStatus::Failed) == "Failed");
}

void test_schwarzschild() {
  INFO("Ray tracing in Schwarzschild spacetime");
  // Trace three equatorial rays in the x-y plane (z=0) around the critical
  // impact parameter in Schwarzschild spacetime, so one ray should fall into
  // the horizon and the others should scatter.
  WrappedGr<gr::Solutions::KerrSchild> background_spacetime{
      {/* mass */ 1., /* spin */ {{0., 0., 0.}}, /* center */ {{0., 0., 0.}}}};
  const double b_critical = sqrt(27.);
  // The delta_b perturbation around the critical impact parameter must be large
  // enough to account for the error we make in starting parallel rays at finite
  // distance from the black hole.
  const double delta_b = 0.5;
  const double distance = 100.;
  // Rays are emitted from top to bottom along the up direction, so their impact
  // parameters are b_critical + 1.5 delta_b, b_critical + 0.5 delta_b, and
  // b_critical - 0.5 delta_b. A ray exactly at the critical impact parameter
  // would orbit the black hole for a long time.
  const double b_center = b_critical + 0.5 * delta_b;
  Camera camera{/* position */
                {distance, b_center, 0.},
                /* focus */ {0., b_center, 0.},
                /* up */ {0., 1., 0.},
                OrthographicProjection{{{0., 2. * delta_b}}},
                /* resolution */ {1, 3},
                /* center_rays */ false,
                /* start_time */ 0.,
                /* interval */ 0.,
                /* num frames */ 1,
                /* integration_time */ 2. * distance};
  const size_t num_rays = camera.num_rays(/* frame */ 0);
  CHECK(num_rays == 3);
  TraceOptions options{};
  options.num_output_steps = 100;
  const RayResults ray_results("ray_results", num_rays);
  const RayStatuses ray_statuses("ray_statuses", num_rays);
  const RayTraces ray_traces("ray_traces", num_rays, options.num_output_steps);
  const std::string output_file_name = "RayTracerTest_Schwarzschild.h5";
  if (file_system::check_if_file_exists(output_file_name)) {
    file_system::rm(output_file_name, true);
  }
  trace_frame(ray_results, ray_statuses, ray_traces,
              make_not_null(&background_spacetime), make_not_null(&camera),
              /* frame */ 0,
              /* rank */ 0, /* num_ranks */ 1, options, output_file_name,
              "Camera/Frame0", "RayTraces/Frame0");
  for (size_t i = 0; i < num_rays; ++i) {
    CAPTURE(i);
    CAPTURE(ray_statuses(i));
    tnsr::I<double, 3> final_position{};
    get<0>(final_position) = ray_results(i, 0);
    get<1>(final_position) = ray_results(i, 1);
    get<2>(final_position) = ray_results(i, 2);
    CAPTURE(final_position);
    // Equatorial rays should keep z = 0
    CHECK(get<2>(final_position) == approx(0.));
    if (i < 2) {
      // Ray above critical impact parameter should scatter
      CHECK(ray_statuses(i) == RayStatus::TimeLimitReached);
      const auto final_position_spherical =
          cartesian_to_spherical(final_position);
      CAPTURE(final_position_spherical);
      CHECK(get<0>(final_position_spherical) > distance / 2.);
      CHECK(get<1>(final_position_spherical) == approx(M_PI_2));
    } else {
      // Ray below critical impact parameter should end near the horizon
      CHECK(ray_statuses(i) == RayStatus::Captured);
      CHECK(radius(ray_results, i) < 2.5);
      CHECK(ray_results(i, num_vars - 1) > options.max_redshift);
    }
  }
  {
    INFO("Check traces");
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      // The first trace step is the initial state of the ray
      const auto initial_state = camera(i, background_spacetime);
      for (size_t d = 0; d < 3; ++d) {
        CHECK(ray_traces(i, 0, d) ==
              get<Tags::Position<double>>(initial_state).get(d));
        CHECK(ray_traces(i, 0, d + 3) ==
              get<Tags::Momentum<double>>(initial_state).get(d));
      }
      CHECK(ray_traces(i, 0, num_vars - 1) == 0.);
      for (size_t k = 0; k < num_vars; ++k) {
        CAPTURE(k);
        const double last_trace =
            ray_traces(i, options.num_output_steps - 1, k);
        if (ray_statuses(i) == RayStatus::TimeLimitReached) {
          // The last trace step is the final state of the ray
          CHECK(last_trace == ray_results(i, k));
        } else {
          // Steps after the ray stopped are NaN
          CHECK(std::isnan(last_trace));
        }
      }
    }
  }
  {
    INFO("Integrate again without intermediate output");
    // Results must be consistent with the run that recorded traces
    options.num_output_steps = 0;
    const RayResults ray_results_no_traces("ray_results_no_traces", num_rays);
    const RayStatuses ray_statuses_no_traces("ray_statuses_no_traces",
                                             num_rays);
    trace_frame(ray_results_no_traces, ray_statuses_no_traces, std::nullopt,
                make_not_null(&background_spacetime), make_not_null(&camera),
                /* frame */ 0,
                /* rank */ 0, /* num_ranks */ 1, options);
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      CHECK(ray_statuses_no_traces(i) == ray_statuses(i));
      for (size_t k = 0; k < num_vars; ++k) {
        CAPTURE(k);
        CHECK(ray_results_no_traces(i, k) == approx(ray_results(i, k)));
      }
    }
  }
  {
    INFO("Stop rays at escape radius");
    options.escape_radius = distance / 2.;
    const RayResults ray_results_escape("ray_results_escape", num_rays);
    const RayStatuses ray_statuses_escape("ray_statuses_escape", num_rays);
    trace_frame(ray_results_escape, ray_statuses_escape, std::nullopt,
                make_not_null(&background_spacetime), make_not_null(&camera),
                /* frame */ 0,
                /* rank */ 0, /* num_ranks */ 1, options);
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      if (i < 2) {
        // Rays start outside the escape radius, but are only stopped once
        // they move outward beyond it after scattering
        CHECK(ray_statuses_escape(i) == RayStatus::Escaped);
        CHECK(radius(ray_results_escape, i) > distance / 2.);
        CHECK(radius(ray_results_escape, i) < radius(ray_results, i));
      } else {
        CHECK(ray_statuses_escape(i) == RayStatus::Captured);
      }
    }
  }
  {
    INFO("Check output file");
    const h5::H5File<h5::AccessType::ReadOnly> output_file(output_file_name);
    const auto& camera_output =
        output_file.get<h5::Dat>("Camera/Frame0").get_data();
    CHECK(camera_output.columns() == num_vars + 2);
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      CHECK(camera_output(i, 0) == static_cast<double>(i));
      CHECK(camera_output(i, 1) ==
            static_cast<double>(static_cast<int>(ray_statuses(i))));
      for (size_t k = 0; k < num_vars; ++k) {
        CAPTURE(k);
        CHECK(camera_output(i, k + 2) == ray_results(i, k));
      }
    }
    output_file.close_current_object();
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      const auto& trace_output =
          output_file.get<h5::Dat>("RayTraces/Frame0/Ray" + std::to_string(i))
              .get_data();
      for (size_t j = 0; j < trace_output.rows(); ++j) {
        CAPTURE(j);
        for (size_t k = 0; k < num_vars; ++k) {
          CAPTURE(k);
          if (std::isnan(ray_traces(i, j, k))) {
            CHECK(std::isnan(trace_output(j, k)));
          } else {
            CHECK(trace_output(j, k) == ray_traces(i, j, k));
          }
        }
      }
      output_file.close_current_object();
    }
  }
  if (file_system::check_if_file_exists(output_file_name)) {
    file_system::rm(output_file_name, true);
  }
}

void test_schwarzschild_shadow() {
  INFO("Shadow of a Schwarzschild black hole");
  // A static observer at radius r sees the shadow of a Schwarzschild black hole
  // with angular radius sin(alpha) = sqrt(27) M / r * sqrt(1 - 2M / r).
  WrappedGr<gr::Solutions::KerrSchild> background_spacetime{
      {/* mass */ 1., /* spin */ {{0., 0., 0.}}, /* center */ {{0., 0., 0.}}}};
  const double distance = 10.;
  const double shadow_angle =
      asin(sqrt(27.) / distance * sqrt(1. - 2. / distance));
  // Scan the vertical direction of the screen across the shadow
  const double half_opening_angle = 1.4 * shadow_angle;
  const size_t resolution = 51;
  Camera camera{
      /* position */
      {distance, 0., 0.},
      /* focus */ {0., 0., 0.},
      /* up */ {0., 0., 1.},
      PerspectiveProjection{{{1., 2. * half_opening_angle * 180. / M_PI}}},
      /* resolution */ {1, resolution},
      /* center_rays */ false,
      /* start_time */ 0.,
      /* interval */ 0.,
      /* num frames */ 1,
      /* integration_time */ 1000.};
  TraceOptions options{};
  options.abs_tol = 1.e-8;
  options.rel_tol = 1.e-8;
  options.escape_radius = 3. * distance;
  const RayResults ray_results("ray_results", resolution);
  const RayStatuses ray_statuses("ray_statuses", resolution);
  trace_frame(ray_results, ray_statuses, std::nullopt,
              make_not_null(&background_spacetime), make_not_null(&camera),
              /* frame */ 0,
              /* rank */ 0, /* num_ranks */ 1, options);
  // The shadow angle must lie between the largest angle of a captured ray and
  // the smallest angle of an escaped ray
  double max_captured_angle = 0.;
  double min_escaped_angle = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < resolution; ++i) {
    CAPTURE(i);
    CAPTURE(ray_statuses(i));
    const double screen_y = 2. * camera.screen_coordinates(i)[1] - 1.;
    const double angle = abs(atan(screen_y * tan(half_opening_angle)));
    CAPTURE(angle);
    if (ray_statuses(i) == RayStatus::Captured) {
      max_captured_angle = std::max(max_captured_angle, angle);
    } else {
      CHECK(ray_statuses(i) == RayStatus::Escaped);
      min_escaped_angle = std::min(min_escaped_angle, angle);
    }
  }
  CAPTURE(shadow_angle);
  CHECK(max_captured_angle < shadow_angle);
  CHECK(min_escaped_angle > shadow_angle);
}

}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.RayTracer",
                  "[Unit][ParallelAlgorithms]") {
  test_partition_ranks();
  test_ray_status_output();
  test_schwarzschild();
  test_schwarzschild_shadow();
}

}  // namespace ray_tracing
