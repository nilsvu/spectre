// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// Need Boost MultiArray because it is used internally by ODEINT
#include "DataStructures/BoostMultiArray.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {
/*!
 * \brief Kernel functor that traces geodesics in parallel
 *
 * This functor runs in a Kokkos parallel for loop and integrates the geodesic
 * equation for each ray in parallel.
 */
template <typename BackgroundSpacetime, typename RaySource>
struct RayTracer {
 private:
  static constexpr size_t Dim = 3;
  using Frame = ::Frame::Inertial;

 public:
  RayTracer(BackgroundSpacetime* background_spacetime, RaySource* ray_source,
            Kokkos::View<double**> ray_results,
            Kokkos::View<double***> ray_traces, const size_t ray_index_start,
            double abs_tol, double rel_tol, size_t num_output_steps,
            ::Verbosity verbosity = ::Verbosity::Silent);

  /// Number of rays to trace on this rank
  size_t num_rays() const { return ray_results_.extent(0); }

  /// Trace the geodesics with rank-local index `i`. The global index of the ray
  /// is `ray_index_start + i`. Can only be called after `initialize()` and the
  /// given index `i` must be smaller than `num_rays()`. This function can be
  /// called from multiple threads running on this rank (node) in parallel.
  KOKKOS_FUNCTION void operator()(const size_t i) const;

  /// Access the final state of each ray after it has been traced.
  /// The first dimension is the rank-local ray index, and the second dimension
  /// contains the ray variables at the end of the trace.
  const auto& ray_results() const { return ray_results_; }

  /// Access the ray variables at `num_output_steps_` points along each ray
  /// after it has been traced. The first dimension is the rank-local ray index,
  /// the second dimension is the output step index along the ray, and the third
  /// dimension contains the ray variables at that point. If the ray terminated
  /// before reaching `num_output_steps_`, the remaining output steps will be
  /// filled with NaNs.
  const auto& ray_traces() const { return ray_traces_; }

 private:
  // Pointers to the background spacetime and ray source
  BackgroundSpacetime* background_spacetime_;
  RaySource* ray_source_;
  // Output data
  Kokkos::View<double**> ray_results_;
  Kokkos::View<double***> ray_traces_;
  // Parameters
  const size_t ray_index_start_;
  double abs_tol_;
  double rel_tol_;
  size_t num_output_steps_;
  ::Verbosity verbosity_{::Verbosity::Silent};
};

/*!
 * \brief Trace geodesics emitted by a ray source through a background
 * spacetime.
 *
 * This function traces many geodesics in parallel using Kokkos and the
 * `RayTracer` functor. Which geodesics to trace is defined by the state of the
 * `ray_source`. Geodesics are partitioned evenly into ranks, with typically one
 * rank per node. Call this function per rank to trace all rays (note that ranks
 * are completely independent).
 * Output data can be optionally written to a file (one per rank).
 *
 * \param ray_results The final state of each ray. Must be sized correctly on
 * input: (num_rays, num_vars)
 * \param ray_traces The state of each ray at the output steps. Unused if
 * `num_output_steps` is 0. Must be sized correctly on input:
 * (num_rays, num_output_steps, num_vars)
 * \param background_spacetime_ptr Background that defines the spacetime
 * geometry.
 * \param ray_source_ptr Ray source that emits the rays to be traced.
 * \param rank The rank number.
 * \param num_ranks Total number of ranks.
 * \param abs_tol Absolute tolerance for the ODE solver.
 * \param rel_tol Relative tolerance for the ODE solver.
 * \param num_output_steps Number of output steps along each ray. If 0, no
 * output steps will be recorded.
 * \param output_file_name Optional name of the output file to write results to.
 * \param results_subfile_name Optional subfile name for the `ray_results`.
 * \param traces_subfile_name Optional subfile name for the `ray_traces`.
 * \param verbosity Verbosity of output.
 */
void trace_frame(
    Kokkos::View<double**> ray_results, Kokkos::View<double***> ray_traces,
    const gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    const gsl::not_null<RaySource*> ray_source_ptr, const size_t frame,
    const size_t rank, const size_t num_ranks, const double abs_tol,
    const double rel_tol, const size_t num_output_steps,
    const std::optional<std::string>& output_file_name = std::nullopt,
    const std::optional<std::string>& results_subfile_name = std::nullopt,
    const std::optional<std::string>& traces_subfile_name = std::nullopt,
    const ::Verbosity verbosity = ::Verbosity::Silent);

/*!
 * \brief Step through frames defined by the ray source and trace each frame
 *
 * For details on the parameters, see `trace_frame`.
 */
void trace_frames(
    const gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    const gsl::not_null<RaySource*> ray_source_ptr, const size_t rank,
    const size_t num_ranks, const double abs_tol, const double rel_tol,
    const size_t num_output_steps,
    const std::optional<std::string>& output_file_name,
    const ::Verbosity verbosity = ::Verbosity::Silent);

}  // namespace ray_tracing
