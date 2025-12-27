// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/RayTracer.hpp"

// Need Boost MultiArray because it is used internally by ODEINT
#include "DataStructures/BoostMultiArray.hpp"

#include <array>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicEquation.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Math.hpp"

namespace ray_tracing {

std::ostream& operator<<(std::ostream& os, const RayStatus status) {
  switch (status) {
    case RayStatus::TimeLimitReached:
      return os << "TimeLimitReached";
    case RayStatus::Escaped:
      return os << "Escaped";
    case RayStatus::Captured:
      return os << "Captured";
    case RayStatus::OutsideBackground:
      return os << "OutsideBackground";
    case RayStatus::Failed:
      return os << "Failed";
    default:
      ERROR("Unknown RayStatus " << static_cast<int>(status));
  }
}

namespace {

// Can switch this to a Variables type by making Variables compatible with Boost
// odeint or by using our own time steppers. Our own time steppers would also
// give us more control over running on GPUs and over loading numeric volume
// data during the integration.
using Vars = std::array<double, num_vars>;

// Thrown by the RHS when the background spacetime is not available at a point.
// Boost odeint provides no other way to abort a step, because it may evaluate
// the RHS at intermediate points of a step. All other termination conditions
// are checked after each step.
struct BackgroundUnavailable : std::exception {};

// RHS of the geodesic equation, called by ODE integrator
template <typename BackgroundSpacetime>
struct GeodesicEquation {
  static constexpr size_t Dim = 3;
  using Frame = ::Frame::Inertial;

  void operator()(const Vars& state, Vars& dt_state, const double time) const {
    // Wrap state and dt_state in tensors. This would be cleaner with a
    // Variables state type.
    const tnsr::I<double, Dim, Frame> position{
        {{state[0], state[1], state[2]}}};
    const tnsr::i<double, Dim, Frame> momentum{
        {{state[3], state[4], state[5]}}};
    const Scalar<double> redshift{state[6]};
    tnsr::I<double, Dim, Frame> dt_position{};
    tnsr::i<double, Dim, Frame> dt_momentum{};
    Scalar<double> dt_redshift{};
    // Evaluate background at the position of the ray
    const auto background_vars = background_spacetime->variables(
        position, time, make_not_null(&block_order));
    if (not background_vars.has_value()) {
      throw BackgroundUnavailable{};
    }
    // Evaluate geodesic equation
    gr::geodesic_equation(
        make_not_null(&dt_position), make_not_null(&dt_momentum),
        make_not_null(&dt_redshift), position, momentum, redshift,
        get<gr::Tags::Lapse<double>>(*background_vars),
        get<::Tags::deriv<gr::Tags::Lapse<double>, tmpl::size_t<Dim>, Frame>>(
            *background_vars),
        get<gr::Tags::Shift<double, Dim, Frame>>(*background_vars),
        get<::Tags::deriv<gr::Tags::Shift<double, Dim, Frame>,
                          tmpl::size_t<Dim>, Frame>>(*background_vars),
        get<gr::Tags::InverseSpatialMetric<double, Dim, Frame>>(
            *background_vars),
        get<::Tags::deriv<gr::Tags::InverseSpatialMetric<double, Dim, Frame>,
                          tmpl::size_t<Dim>, Frame>>(*background_vars),
        get<gr::Tags::ExtrinsicCurvature<double, Dim, Frame>>(
            *background_vars));
    // Unwrap dt_state back into the state array
    dt_state[0] = get<0>(dt_position);
    dt_state[1] = get<1>(dt_position);
    dt_state[2] = get<2>(dt_position);
    dt_state[3] = get<0>(dt_momentum);
    dt_state[4] = get<1>(dt_momentum);
    dt_state[5] = get<2>(dt_momentum);
    dt_state[6] = get(dt_redshift);
  }

  const BackgroundSpacetime* background_spacetime;
  // Cached block order for speeding up numeric volume data interpolations
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::vector<size_t> block_order{};
};

double coordinate_radius(const Vars& state) {
  return sqrt(square(state[0]) + square(state[1]) + square(state[2]));
}

// Integrate a single ray from `start_time` for `integration_time` (which can be
// negative to integrate backwards in time) until a termination condition is
// met. On input, `ray` holds the initial state. On output, it holds the state
// at the last completed step. If `ray_traces` is provided, records the state of
// the ray at `options.num_output_steps` uniformly spaced times into the
// `ray_traces` at index `trace_index`.
//
// Throws `BackgroundUnavailable` if the background spacetime is not available
// at a point where the integrator evaluates the RHS.
template <typename BackgroundSpacetime>
RayStatus integrate_ray(const gsl::not_null<Vars*> ray, const double start_time,
                        const double integration_time,
                        const BackgroundSpacetime& background_spacetime,
                        const TraceOptions& options,
                        const std::optional<RayTraces>& ray_traces,
                        const size_t trace_index) {
  namespace odeint = boost::numeric::odeint;
  using OdeState = odeint::runge_kutta_dopri5<Vars>;
  auto stepper =
      odeint::make_dense_output(options.abs_tol, options.rel_tol, OdeState{});
  const GeodesicEquation<BackgroundSpacetime> rhs{&background_spacetime};
  const double end_time = start_time + integration_time;
  // Sign of the integration direction in time
  const double direction = integration_time < 0.0 ? -1.0 : 1.0;

  // Output steps are spaced uniformly in time. The last output step is at the
  // end time.
  const size_t num_output_steps = options.num_output_steps;
  size_t num_recorded = 0;
  const auto output_time = [&](const size_t step) {
    return step + 1 == num_output_steps
               ? end_time
               : start_time + static_cast<double>(step) * integration_time /
                                  static_cast<double>(num_output_steps - 1);
  };
  const auto record = [&](const Vars& state) {
    for (size_t k = 0; k < num_vars; ++k) {
      ray_traces.value()(trace_index, num_recorded, k) = gsl::at(state, k);
    }
    ++num_recorded;
  };
  if (num_output_steps > 0) {
    record(*ray);
  }

  // There are probably smarter ways to choose the initial time step, but this
  // seems to work well enough and the adaptive stepper will take over anyway
  const double initial_dt =
      sqrt(options.abs_tol + options.rel_tol * abs(integration_time)) *
      direction;
  stepper.initialize(*ray, start_time, initial_dt);
  double previous_radius = coordinate_radius(*ray);
  while (direction * (end_time - stepper.current_time()) > 0.0) {
    // Don't step past the end time, so the background spacetime is only
    // evaluated within the time bounds of the ray source
    if (direction *
            (stepper.current_time() + stepper.current_time_step() - end_time) >
        0.0) {
      const Vars current_state = stepper.current_state();
      stepper.initialize(current_state, stepper.current_time(),
                         end_time - stepper.current_time());
    }
    stepper.do_step(std::cref(rhs));
    *ray = stepper.current_state();
    // Record output steps that were covered by this step
    while (num_recorded < num_output_steps and
           direction * (output_time(num_recorded) - stepper.current_time()) <=
               0.0) {
      Vars output_state{};
      stepper.calc_state(output_time(num_recorded), output_state);
      record(output_state);
    }
    // Check termination conditions
    if ((*ray)[num_vars - 1] > options.max_redshift) {
      return RayStatus::Captured;
    }
    if (options.escape_radius.has_value()) {
      const double radius = coordinate_radius(*ray);
      if (radius > options.escape_radius.value() and radius > previous_radius) {
        return RayStatus::Escaped;
      }
      previous_radius = radius;
    }
  }
  // The last output step may not have been recorded yet due to roundoff in the
  // time
  while (num_recorded < num_output_steps) {
    record(*ray);
  }
  return RayStatus::TimeLimitReached;
}

}  // namespace

template <typename BackgroundSpacetime, typename RaySource>
RayTracer<BackgroundSpacetime, RaySource>::RayTracer(
    BackgroundSpacetime* background_spacetime, RaySource* ray_source,
    RayResults ray_results, RayStatuses ray_statuses,
    std::optional<RayTraces> ray_traces, const size_t ray_index_start,
    TraceOptions options)
    : background_spacetime_(background_spacetime),
      ray_source_(ray_source),
      ray_results_(std::move(ray_results)),
      ray_statuses_(std::move(ray_statuses)),
      ray_traces_(std::move(ray_traces)),
      ray_index_start_(ray_index_start),
      options_(std::move(options)) {}

template <typename BackgroundSpacetime, typename RaySource>
void RayTracer<BackgroundSpacetime, RaySource>::operator()(
    const size_t i) const {
  // Initialize ray
  const size_t ray_index = ray_index_start_ + i;
  const auto initial_state = (*ray_source_)(ray_index, *background_spacetime_);
  const auto& initial_position =
      get<Tags::Position<double, Dim, Frame>>(initial_state);
  const auto& initial_momentum =
      get<Tags::Momentum<double, Dim, Frame>>(initial_state);
  Vars ray{{get<0>(initial_position), get<1>(initial_position),
            get<2>(initial_position), get<0>(initial_momentum),
            get<1>(initial_momentum), get<2>(initial_momentum), 0.0}};
  const double start_time = get<::Tags::Time>(initial_state);
  // Integration time can be negative if tracing backwards in time
  const double integration_time = get<Tags::IntegrationTime>(initial_state);

  // Fill ray traces with NaNs so steps after the ray stopped are marked
  if (options_.num_output_steps > 0) {
    ASSERT(options_.num_output_steps >= 2,
           "num_output_steps must be at least 2 to record intermediate steps.");
    ASSERT(ray_traces_.has_value(),
           "ray_traces must be provided if num_output_steps > 0.");
    for (size_t j = 0; j < options_.num_output_steps; ++j) {
      for (size_t k = 0; k < num_vars; ++k) {
        ray_traces_.value()(i, j, k) =
            std::numeric_limits<double>::signaling_NaN();
      }
    }
  }

  // Integrate the geodesic. Exceptions must not escape the parallel loop, so
  // catch them here and record the status of the ray.
  RayStatus status = RayStatus::Failed;
  try {
    status = integrate_ray(make_not_null(&ray), start_time, integration_time,
                           *background_spacetime_, options_, ray_traces_, i);
  } catch (const BackgroundUnavailable& /*e*/) {
    status = RayStatus::OutsideBackground;
  } catch (const std::exception& e) {
    // This is an actual error, so print it unless output is silenced
    status = RayStatus::Failed;
    if (options_.verbosity >= ::Verbosity::Quiet) {
      Parallel::printf("Integrating ray %zu failed: %s\n", ray_index, e.what());
    }
  }
  if (options_.verbosity >= ::Verbosity::Debug) {
    Parallel::printf("Done integrating ray %zu: %s\n", ray_index,
                     get_output(status));
  }
  ray_statuses_(i) = status;
  for (size_t j = 0; j < num_vars; ++j) {
    ray_results_(i, j) = gsl::at(ray, j);
  }
}

std::pair<size_t, size_t> partition_ranks(const size_t total_num_rays,
                                          const size_t rank,
                                          const size_t num_ranks) {
  ASSERT(rank < num_ranks,
         "Rank " << rank << " is out of range for " << num_ranks << " ranks.");
  // Partition the rays evenly across ranks
  const size_t ray_index_start = total_num_rays * rank / num_ranks;
  const size_t ray_index_end = total_num_rays * (rank + 1) / num_ranks;
  return {ray_index_start, ray_index_end - ray_index_start};
}

void trace_frame(
    const RayResults& ray_results, const RayStatuses& ray_statuses,
    const std::optional<RayTraces>& ray_traces,
    const gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    const gsl::not_null<RaySource*> ray_source_ptr, const size_t frame,
    const size_t rank, const size_t num_ranks, const TraceOptions& options,
    const std::optional<std::string>& output_file_name,
    const std::optional<std::string>& results_subfile_name,
    const std::optional<std::string>& traces_subfile_name) {
  auto& background_spacetime = *background_spacetime_ptr;
  auto& ray_source = *ray_source_ptr;
  const size_t num_output_steps = options.num_output_steps;

  // Load data for this frame
  background_spacetime.initialize(ray_source.time_bounds(frame));

  // Initialize ray source given the background spacetime
  ray_source.initialize(frame, background_spacetime);

  // Get number of rays to trace for this rank
  const auto [ray_index_start, num_rays_this_rank] =
      partition_ranks(ray_source.num_rays(frame), rank, num_ranks);

  // Check size of output arrays
  ASSERT(ray_results.extent(0) == num_rays_this_rank,
         "ray_results has incorrect length. Expected "
             << num_rays_this_rank << " but got " << ray_results.extent(0));
  ASSERT(ray_statuses.extent(0) == num_rays_this_rank,
         "ray_statuses has incorrect length. Expected "
             << num_rays_this_rank << " but got " << ray_statuses.extent(0));
  if (num_output_steps > 0) {
    ASSERT(ray_traces.has_value(),
           "ray_traces must be provided if num_output_steps > 0.");
    ASSERT(ray_traces->extent(0) == num_rays_this_rank,
           "ray_traces has incorrect length (first dim). Expected "
               << num_rays_this_rank << " but got " << ray_traces->extent(0));
    ASSERT(ray_traces->extent(1) == num_output_steps,
           "ray_traces has incorrect number of output steps (second dim). "
           "Expected "
               << num_output_steps << " but got " << ray_traces->extent(1));
  }

  // Integrate geodesics in parallel
  if (options.verbosity >= ::Verbosity::Quiet) {
    Parallel::printf("Rank %zu: tracing %zu geodesics on %d threads...\n", rank,
                     num_rays_this_rank,
                     Kokkos::DefaultHostExecutionSpace{}.concurrency());
  }
  // Rays take vastly different amounts of time to trace (e.g. rays that orbit
  // the black hole near the photon sphere take much longer than rays that
  // escape), so schedule them dynamically
  const Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Schedule<Kokkos::Dynamic>,
                            Kokkos::IndexType<size_t>>
      policy{0, num_rays_this_rank};
  // Unwrap dynamic types because Kokkos and generally GPU code is not super
  // happy with polymorphism
  call_with_dynamic_type<void, all_background_spacetimes>(
      &background_spacetime,
      [&](const auto* const background_spacetime_derived) {
        call_with_dynamic_type<void, all_ray_sources>(
            &ray_source, [&](const auto* const ray_source_derived) {
              const RayTracer ray_tracer(
                  background_spacetime_derived, ray_source_derived, ray_results,
                  ray_statuses, ray_traces, ray_index_start, options);
              Kokkos::parallel_for("Trace geodesics", policy, ray_tracer);
            });
      });

  // Write results to disk if requested
  if (output_file_name.has_value()) {
    h5::H5File<h5::AccessType::ReadWrite> h5file(output_file_name.value(),
                                                 true);
    const std::vector<std::string> vars_legend{"X",  "Y",  "Z",       "Px",
                                               "Py", "Pz", "Redshift"};
    ASSERT(vars_legend.size() == num_vars,
           "Legend size does not match number of ray variables.");
    // Write frame data
    if (results_subfile_name.has_value()) {
      std::vector<std::string> legend{"RayIndex", "Status"};
      legend.insert(legend.end(), vars_legend.begin(), vars_legend.end());
      auto& frame_datfile =
          h5file.insert<h5::Dat>(results_subfile_name.value(), legend);
      std::vector<double> row_data(legend.size());
      for (size_t i = 0; i < num_rays_this_rank; ++i) {
        row_data[0] = static_cast<double>(ray_index_start + i);
        row_data[1] = static_cast<double>(static_cast<int>(ray_statuses(i)));
        for (size_t k = 0; k < num_vars; ++k) {
          row_data[k + 2] = ray_results(i, k);
        }  // for vars
        frame_datfile.append(row_data);
      }  // for num_rays_this_rank
      h5file.close_current_object();
    }
    // Write ray traces
    if (num_output_steps > 0 and traces_subfile_name.has_value()) {
      std::vector<double> row_data(num_vars);
      for (size_t i = 0; i < num_rays_this_rank; ++i) {
        auto& datfile =
            h5file.insert<h5::Dat>(traces_subfile_name.value() + "/Ray" +
                                       std::to_string(ray_index_start + i),
                                   vars_legend);
        for (size_t j = 0; j < num_output_steps; ++j) {
          for (size_t k = 0; k < num_vars; ++k) {
            row_data[k] = (*ray_traces)(i, j, k);
          }  // for vars
          datfile.append(row_data);
        }  // for num_output_steps
        h5file.close_current_object();
      }  // for num_rays_this_rank
    }
  }  // if output_file_name
}

void trace_frames(
    const gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    const gsl::not_null<RaySource*> ray_source_ptr, const size_t rank,
    const size_t num_ranks, const TraceOptions& options,
    const std::optional<std::string>& output_file_name) {
  // Ray source keeps track of time
  auto& ray_source = *ray_source_ptr;
  // Allocate memory for output (will be resized later)
  RayResults ray_results;
  RayStatuses ray_statuses;
  std::optional<RayTraces> ray_traces;
  // Loop over frames
  const size_t num_frames = ray_source.num_frames();
  Kokkos::Timer timer{};
  for (size_t frame = 0; frame < num_frames; ++frame) {
    if (options.verbosity >= ::Verbosity::Quiet) {
      Parallel::printf("Rank %zu: tracing frame %zu...\n", rank, frame);
    }
    timer.reset();
    // Resize output arrays if needed
    const size_t num_rays_this_rank =
        partition_ranks(ray_source.num_rays(frame), rank, num_ranks).second;
    if (ray_results.extent(0) != num_rays_this_rank) {
      ray_results = RayResults("RayResults", num_rays_this_rank);
      ray_statuses = RayStatuses("RayStatuses", num_rays_this_rank);
      if (options.num_output_steps > 0) {
        ray_traces = RayTraces("RayTraces", num_rays_this_rank,
                               options.num_output_steps);
      }
    }
    // Trace geodesics for this frame
    trace_frame(ray_results, ray_statuses, ray_traces, background_spacetime_ptr,
                ray_source_ptr, frame, rank, num_ranks, options,
                output_file_name, "Camera/Frame" + std::to_string(frame),
                "RayTraces/Frame" + std::to_string(frame));
    if (options.verbosity >= ::Verbosity::Quiet) {
      Parallel::printf("Rank %zu: frame %zu done in %g seconds.\n", rank, frame,
                       timer.seconds());
    }
  }
}

}  // namespace ray_tracing
