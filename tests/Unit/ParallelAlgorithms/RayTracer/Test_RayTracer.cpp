// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/CartesianToSpherical.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"
#include "ParallelAlgorithms/RayTracer/RayTracer.hpp"

namespace ray_tracing {

void test_schwarzschild() {
  INFO("Ray tracing in Schwarzschild spacetime");
  // Trace two equatorial rays in the x-y plane (z=0) around the critical impact
  // parameter in Schwarzschild spacetime, so one ray should fall into the
  // horizon and the other should scatter.
  WrappedGr<gr::Solutions::KerrSchild> background_spacetime{
      {/* mass */ 1., /* spin */ {{0., 0., 0.}}}};
  const double b_critical = sqrt(27.);
  // The delta_b perturbation around the critical impact parameter must be large
  // enough to account for the error we make in starting parallel rays at finite
  // distance from the black hole.
  const double delta_b = 0.5;
  const double distance = 100.;
  Camera camera{/* position */ {{distance, b_critical, 0.}},
                /* focus */ {{0., b_critical, 0.}},
                /* up */ {{0., 1., 0.}},
                /* parallel_rays */ true,
                /* opening_angle */ delta_b,
                /* resolution */ {{1, 3}},
                /* time_range */ {{0., 0.}},
                /* interval */ 0.,
                /* integration_time */ 2. * distance};
  const size_t num_rays = camera.num_rays(/* frame */ 0);
  const size_t num_vars = 7;
  const size_t num_output_steps = static_cast<size_t>(2. * distance);
  Kokkos::View<double**> ray_results("ray_results", num_rays, num_vars);
  Kokkos::View<double***> ray_traces("ray_traces", num_rays, num_output_steps,
                                     num_vars);
  std::string output_file_name = "RayTracerTest_Schwarzschild.h5";
  if (file_system::check_if_file_exists(output_file_name)) {
    file_system::rm(output_file_name, true);
  }
  trace_frame(ray_results, ray_traces, make_not_null(&background_spacetime),
              make_not_null(&camera),
              /* frame */ 0,
              /* rank */ 0, /* num_ranks */ 1, /* abs_tol */ 1e-6,
              /*rel_tol*/ 1e-6, /* num_output_steps */ num_output_steps,
              output_file_name, "Camera/Frame0", "RayTraces/Frame0");
  for (size_t i = 0; i < num_rays; ++i) {
    CAPTURE(i);
    tnsr::I<double, 3> final_position{};
    get<0>(final_position) = ray_results(i, 0);
    get<1>(final_position) = ray_results(i, 1);
    get<2>(final_position) = ray_results(i, 2);
    CAPTURE(final_position);
    if (i == 0) {
      // Ray below critical impact parameter should end at the horizon
      CHECK(std::isnan(get<0>(final_position)));
      CHECK(std::isnan(get<1>(final_position)));
      CHECK(std::isnan(get<2>(final_position)));
    } else {
      /// Ray above critical impact parameter should scatter
      const auto final_position_spherical =
          cartesian_to_spherical(final_position);
      CAPTURE(final_position_spherical);
      CHECK(get<0>(final_position_spherical) > distance / 2.);
      // Equatorial rays should keep z = 0
      CHECK(get<2>(final_position) == approx(0.));
      CHECK(get<1>(final_position_spherical) == approx(M_PI_2));
    }
  }
  // Check traces
  {
    // Make sure the final state of the rays matches the ray results
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      for (size_t k = 0; k < num_vars; ++k) {
        CAPTURE(k);
        if (std::isnan(ray_results(i, k))) {
          CHECK(std::isnan(ray_traces(i, num_output_steps - 1, k)));
        } else {
          CHECK(ray_traces(i, num_output_steps - 1, k) == ray_results(i, k));
        }
      }
    }
  }
  // Check output file
  {
    const h5::H5File<h5::AccessType::ReadOnly> output_file(output_file_name);
    const auto& camera_output =
        output_file.get<h5::Dat>("Camera/Frame0").get_data();
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      for (size_t k = 0; k < num_vars; ++k) {
        CAPTURE(k);
        if (std::isnan(camera_output(i, k))) {
          CHECK(std::isnan(ray_results(i, k)));
        } else {
          CHECK(camera_output(i, k) == ray_results(i, k));
        }
      }
    }
    output_file.close_current_object();
    for (size_t i = 0; i < num_rays; ++i) {
      CAPTURE(i);
      const auto& trace_output =
          output_file.get<h5::Dat>("RayTraces/Frame0/Ray" + std::to_string(i))
              .get_data();
      for (size_t j = 0; j < num_output_steps; ++j) {
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

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.RayTracer",
                  "[Unit][ParallelAlgorithms]") {
  test_schwarzschild();
}

}  // namespace ray_tracing
