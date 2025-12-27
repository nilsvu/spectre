// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RayTracer.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/PrettyType.hpp"

namespace py = pybind11;

namespace ray_tracing {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  if (not Kokkos::is_initialized()) {
    Kokkos::initialize();
    // Finalize Kokkos when the interpreter exits. All Kokkos views are local
    // to the bound functions, so none outlive this point.
    py::module_::import("atexit").attr("register")(
        py::cpp_function([]() { Kokkos::finalize(); }));
  }
  py::class_<BackgroundSpacetime>(m, "BackgroundSpacetime").doc() =
      "Abstract base class for background spacetimes";
  py::class_<WrappedGr<gr::Solutions::KerrSchild>, BackgroundSpacetime>(
      m, "KerrSchild")
      .def(py::init([](const double mass, const std::array<double, 3> spin) {
             return WrappedGr<gr::Solutions::KerrSchild>{
                 gr::Solutions::KerrSchild{mass, spin, {0., 0., 0.}}};
           }),
           py::arg("mass"), py::arg("spin"));
  py::class_<RaySource>(m, "RaySource").doc() =
      "Abstract base class for ray sources";
  py::class_<PerspectiveProjection>(m, "PerspectiveProjection",
                                    PerspectiveProjection::help)
      .def(py::init<std::array<double, 2>>(), py::arg("opening_angle"))
      .def_readonly("opening_angle", &PerspectiveProjection::opening_angle);
  py::class_<OrthographicProjection>(m, "OrthographicProjection",
                                     OrthographicProjection::help)
      .def(py::init<std::array<double, 2>>(), py::arg("extent"))
      .def_readonly("extent", &OrthographicProjection::extent);
  py::class_<Camera, RaySource>(m, "Camera")
      .def(py::init<std::array<double, 3>, std::array<double, 3>,
                    std::array<double, 3>, Camera::Projection::type,
                    std::array<size_t, 2>, bool, double, double, size_t, double,
                    bool>(),
           py::arg("position"), py::arg("focus"), py::arg("up"),
           py::arg("projection"), py::arg("resolution"), py::arg("center_rays"),
           py::arg("start_time"), py::arg("interval"), py::arg("num_frames"),
           py::arg("integration_time"), py::arg("only_upper_half") = false)
      .def_property_readonly("position", &Camera::position)
      .def_property_readonly("four_velocity", &Camera::four_velocity)
      .def_property_readonly("direction", &Camera::direction)
      .def_property_readonly("up", &Camera::up)
      .def_property_readonly("right", &Camera::right)
      .def_property_readonly("projection", &Camera::projection)
      .def_property_readonly("resolution", &Camera::resolution)
      .def_property_readonly("center_rays", &Camera::center_rays)
      .def_property_readonly("start_time", &Camera::start_time)
      .def_property_readonly("interval", &Camera::interval)
      .def_property_readonly("integration_time", &Camera::integration_time)
      .def_property_readonly("only_upper_half", &Camera::only_upper_half)
      .def_property_readonly("time", &Camera::time);
  py::enum_<RayStatus>(m, "RayStatus", "Why the integration of a ray stopped")
      .value("TimeLimitReached", RayStatus::TimeLimitReached)
      .value("Escaped", RayStatus::Escaped)
      .value("Captured", RayStatus::Captured)
      .value("OutsideBackground", RayStatus::OutsideBackground)
      .value("Failed", RayStatus::Failed);
  const TraceOptions default_options{};
  m.def(
      "trace_frame",
      [](BackgroundSpacetime& background_spacetime, RaySource& ray_source,
         const size_t frame, const size_t rank, const size_t num_ranks,
         const double abs_tol, const double rel_tol,
         const size_t num_output_steps,
         const std::optional<double> escape_radius, const double max_redshift,
         const std::optional<std::string>& output_file_name,
         const std::optional<std::string>& results_subfile_name,
         const std::optional<std::string>& traces_subfile_name) {
        const TraceOptions options{.abs_tol = abs_tol,
                                   .rel_tol = rel_tol,
                                   .num_output_steps = num_output_steps,
                                   .escape_radius = escape_radius,
                                   .max_redshift = max_redshift};
        const size_t num_rays_this_rank =
            partition_ranks(ray_source.num_rays(frame), rank, num_ranks).second;
        const RayResults ray_results{"RayResults", num_rays_this_rank};
        const RayStatuses ray_statuses{"RayStatuses", num_rays_this_rank};
        std::optional<RayTraces> ray_traces;
        if (num_output_steps > 0) {
          ray_traces.emplace("RayTraces", num_rays_this_rank, num_output_steps);
        }
        trace_frame(ray_results, ray_statuses, ray_traces,
                    make_not_null(&background_spacetime),
                    make_not_null(&ray_source), frame, rank, num_ranks, options,
                    output_file_name, results_subfile_name,
                    traces_subfile_name);
        // Copies the data because no base object is passed
        const py::array_t<double> results{
            {ray_results.extent(0), ray_results.extent(1)}, ray_results.data()};
        py::array_t<int> statuses(static_cast<py::ssize_t>(num_rays_this_rank));
        auto statuses_view = statuses.mutable_unchecked<1>();
        for (size_t i = 0; i < num_rays_this_rank; ++i) {
          statuses_view(static_cast<py::ssize_t>(i)) =
              static_cast<int>(ray_statuses(i));
        }
        return py::make_tuple(results, statuses);
      },
      py::arg("background_spacetime"), py::arg("ray_source"),
      py::arg("frame") = 0, py::arg("rank") = 0, py::arg("num_ranks") = 1,
      py::arg("abs_tol") = default_options.abs_tol,
      py::arg("rel_tol") = default_options.rel_tol,
      py::arg("num_output_steps") = default_options.num_output_steps,
      py::arg("escape_radius") = default_options.escape_radius,
      py::arg("max_redshift") = default_options.max_redshift,
      py::arg("output_file_name") = std::nullopt,
      py::arg("results_subfile_name") = std::nullopt,
      py::arg("traces_subfile_name") = std::nullopt,
      "Trace the rays of a frame on this rank. Returns the final state of each "
      "ray (an array of shape (num_rays, 7) with columns X, Y, Z, Px, Py, Pz, "
      "Redshift) and why each ray stopped (an array of RayStatus values).");
}

}  // namespace ray_tracing
