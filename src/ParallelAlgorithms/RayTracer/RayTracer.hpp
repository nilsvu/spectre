// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// Functionality for tracing rays (geodesics) in a background spacetime
namespace ray_tracing {

/// Number of variables that describe the state of a ray: position (3),
/// momentum (3), and redshift (1). See `gr::geodesic_equation`.
constexpr size_t num_vars = 7;

/// Why the integration of a ray stopped
enum class RayStatus : int {
  /// The ray was integrated for its full integration time
  TimeLimitReached = 0,
  /// The ray moved outward beyond the `TraceOptions::escape_radius`
  Escaped = 1,
  /// The redshift variable exceeded the `TraceOptions::max_redshift`, which
  /// indicates that the ray approached a horizon
  Captured = 2,
  /// The ray reached a point where the background spacetime is not available,
  /// e.g. outside the domain of numeric data or inside an excision region
  OutsideBackground = 3,
  /// The integration failed with an error
  Failed = 4
};

std::ostream& operator<<(std::ostream& os, RayStatus status);

/// Parameters that control how rays are traced
struct TraceOptions {
  /// Absolute tolerance of the adaptive ODE integrator
  double abs_tol = 1.e-6;
  /// Relative tolerance of the adaptive ODE integrator
  double rel_tol = 1.e-6;
  /// Number of output steps along each ray, uniformly spaced in time. If 0,
  /// only the final state of each ray is recorded. Otherwise, must be at
  /// least 2.
  size_t num_output_steps = 0;
  /// Rays that move outward beyond this coordinate distance from the origin
  /// stop with status `RayStatus::Escaped`. Set this to the radius of the
  /// "celestial sphere" to stop tracing rays that have escaped.
  std::optional<double> escape_radius = std::nullopt;
  /// Rays stop with status `RayStatus::Captured` when the redshift variable
  /// \f$\ln(\alpha p^0)\f$ exceeds this value. Rays that are traced backwards
  /// in time from a camera and approach a black hole horizon are strongly
  /// blueshifted, so this condition stops them near the horizon.
  double max_redshift = 5.0;
  ::Verbosity verbosity = ::Verbosity::Silent;
};

/// Final state of each ray, indexed by (ray, var). Lives in host memory
/// because the ray tracer currently runs on the host.
using RayResults = Kokkos::View<double* [num_vars], Kokkos::HostSpace>;

/// Why each ray stopped, indexed by ray.
using RayStatuses = Kokkos::View<RayStatus*, Kokkos::HostSpace>;

/// State of each ray at the output steps, indexed by (ray, step, var). Lives in
/// host memory because the ray tracer currently runs on the host.
using RayTraces = Kokkos::View<double** [num_vars], Kokkos::HostSpace>;

/*!
 * \brief Functor that traces geodesics in parallel
 *
 * This functor can run in a Kokkos parallel for loop on a host execution space
 * to integrate the geodesic equation for each ray in parallel. It can't run on
 * devices (GPUs) yet because the ODE integrator, the geodesic equation, and the
 * background spacetimes are host-only.
 *
 * Each ray is integrated until one of the conditions listed in `RayStatus` is
 * met.
 */
template <typename BackgroundSpacetime, typename RaySource>
struct RayTracer {
 private:
  static constexpr size_t Dim = 3;
  using Frame = ::Frame::Inertial;

 public:
  RayTracer(BackgroundSpacetime* background_spacetime, RaySource* ray_source,
            RayResults ray_results, RayStatuses ray_statuses,
            std::optional<RayTraces> ray_traces, size_t ray_index_start,
            TraceOptions options);

  /// Number of rays to trace on this rank (node)
  size_t num_rays() const { return ray_results_.extent(0); }

  /// Trace the ray with rank-local index `i`. The given index must be smaller
  /// than `num_rays()`. The global index of the ray is `ray_index_start + i`.
  /// This function can be called from multiple threads running on this rank
  /// (node) in parallel.
  void operator()(size_t i) const;

  /// Access the final state of each ray after it has been traced.
  /// The first dimension is the rank-local ray index, and the second dimension
  /// contains the ray variables where the integration stopped.
  const auto& ray_results() const { return ray_results_; }

  /// Access why the integration of each ray stopped.
  const auto& ray_statuses() const { return ray_statuses_; }

  /// Access the ray variables at `num_output_steps` points along each ray
  /// after it has been traced. The first dimension is the rank-local ray index,
  /// the second dimension is the output step index along the ray, and the third
  /// dimension contains the ray variables at that point. If the ray stopped
  /// before its full integration time, the remaining output steps will be
  /// filled with NaNs.
  const auto& ray_traces() const { return ray_traces_; }

 private:
  // Pointers to the background spacetime and ray source
  BackgroundSpacetime* background_spacetime_;
  RaySource* ray_source_;
  // Output data
  RayResults ray_results_;
  RayStatuses ray_statuses_;
  std::optional<RayTraces> ray_traces_;
  // Parameters
  size_t ray_index_start_;
  TraceOptions options_;
};

/// Returns the global start index and the number of rays for a given rank
/// (node), partitioning the total number of rays evenly across ranks. The
/// number of rays on any two ranks differs by at most one.
std::pair<size_t, size_t> partition_ranks(size_t total_num_rays, size_t rank,
                                          size_t num_ranks);

/*!
 * \brief Trace geodesics emitted by a ray source through a background
 * spacetime.
 *
 * This function traces many geodesics in parallel using Kokkos and the
 * `RayTracer` functor. Which geodesics to trace is defined by the state of the
 * `ray_source`. Geodesics are partitioned evenly across ranks (nodes). Call
 * this function once per rank to trace all rays on that rank (note that ranks
 * are completely independent). Output data can be optionally written to a file
 * (one per rank).
 *
 * \param ray_results The final state of each ray on this rank. Must be sized
 * correctly on input: (num_rays, num_vars)
 * \param ray_statuses Why each ray on this rank stopped. Must be sized
 * correctly on input: (num_rays)
 * \param ray_traces The state of each ray at the output steps. Unused if
 * `options.num_output_steps` is 0. Must be sized correctly on input:
 * (num_rays, num_output_steps, num_vars)
 * \param background_spacetime_ptr Background that defines the spacetime
 * geometry.
 * \param ray_source_ptr Ray source that emits the rays to be traced.
 * \param frame Index of the frame to trace (see `RaySource`).
 * \param rank Rank index.
 * \param num_ranks Total number of ranks.
 * \param options Parameters that control how rays are traced.
 * \param output_file_name Optional name of the output file to write results to.
 * \param results_subfile_name Optional subfile name for the `ray_results` and
 * `ray_statuses`.
 * \param traces_subfile_name Optional subfile name for the `ray_traces`.
 */
void trace_frame(
    const RayResults& ray_results, const RayStatuses& ray_statuses,
    const std::optional<RayTraces>& ray_traces,
    gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    gsl::not_null<RaySource*> ray_source_ptr, size_t frame, size_t rank,
    size_t num_ranks, const TraceOptions& options,
    const std::optional<std::string>& output_file_name = std::nullopt,
    const std::optional<std::string>& results_subfile_name = std::nullopt,
    const std::optional<std::string>& traces_subfile_name = std::nullopt);

/*!
 * \brief Step through frames defined by the ray source and trace each frame
 *
 * For details on the parameters, see `trace_frame`.
 */
void trace_frames(
    gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    gsl::not_null<RaySource*> ray_source_ptr, size_t rank, size_t num_ranks,
    const TraceOptions& options,
    const std::optional<std::string>& output_file_name = std::nullopt);

}  // namespace ray_tracing
