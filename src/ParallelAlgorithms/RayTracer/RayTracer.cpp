// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/RayTracer.hpp"

#include <array>
#include <boost/numeric/odeint.hpp>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicEquation.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace ray_tracing {

namespace {

struct MemoryObserver {
  using Vars = std::array<double, 7>;
  void operator()(const Vars& state, const double /*time*/) {
    data.push_back(state);
  }
  std::vector<Vars> data;
};

struct KokkosObserver {
  using Vars = std::array<double, 7>;
  void operator()(const Vars& state, const double /*time*/) {
    for (size_t i = 0; i < state.size(); ++i) {
      data(step, i) = state[i];
    }
    ++step;
  }
  Kokkos::View<double* [7]> data;
  size_t step = 0;
};

struct StreamingObserver {
  std::ostream& m_out;

  StreamingObserver(std::ostream& out) : m_out(out) {}

  template <class State>
  void operator()(const State& state, double time) const {
    m_out << time;
    for (size_t i = 0; i < state.size(); ++i)
      m_out << "\t" << state[i];
    m_out << "\n";
  }
};

}  // namespace

template <typename BackgroundSpacetime, typename RaySource>
RayTracer<BackgroundSpacetime, RaySource>::RayTracer(
    BackgroundSpacetime* background_spacetime, RaySource* ray_source,
    Kokkos::View<double**> ray_results, Kokkos::View<double***> ray_traces,
    const size_t ray_index_start, const double abs_tol, const double rel_tol,
    const size_t num_output_steps, const ::Verbosity verbosity)
    : background_spacetime_(background_spacetime),
      ray_source_(ray_source),
      ray_results_(ray_results),
      ray_traces_(ray_traces),
      ray_index_start_(ray_index_start),
      abs_tol_(abs_tol),
      rel_tol_(rel_tol),
      num_output_steps_(num_output_steps),
      verbosity_(verbosity) {}

template <typename BackgroundSpacetime, typename RaySource>
KOKKOS_FUNCTION void RayTracer<BackgroundSpacetime, RaySource>::operator()(
    size_t i) const {
  // Can switch this to a Variables type by making Variables compatible with
  // Boost odeint or by using our own time steppers. Our own time steppers would
  // also give us more control over running on GPUs, over loading numeric
  // volume data during the integration, and over termination conditions.
  using Vars = std::array<double, 7>;

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
  // There are probably smarter ways to choose the initial time step, but this
  // seems to work well enough and the adaptive stepper will take over anyway
  const double initial_dt =
      sqrt(abs_tol_ + rel_tol_ * abs(integration_time)) * sgn(integration_time);

  // Set up ODE integrator
  using OdeState = boost::numeric::odeint::runge_kutta_dopri5<Vars>;
  using OdeIntegrator = boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<OdeState>>;
  OdeIntegrator stepper = make_dense_output(abs_tol_, rel_tol_, OdeState{});

  // Set up observation times
  const double observer_dt =
      integration_time / static_cast<double>(num_output_steps_ - 1);
  std::vector<double> observer_times(num_output_steps_);
  for (size_t j = 0; j < num_output_steps_; ++j) {
    observer_times[j] = start_time + j * observer_dt;
  }

  // Set up observer to record intermediate ray states
  const auto ray_trace_subview =
      Kokkos::subview(ray_traces_, i, Kokkos::ALL, Kokkos::ALL);
  for (size_t j = 0; j < num_output_steps_; ++j) {
    for (size_t k = 0; k < ray.size(); ++k) {
      ray_trace_subview(j, k) = std::numeric_limits<double>::signaling_NaN();
    }
  }
  KokkosObserver observer{ray_trace_subview};

  // Set up block order caching for numeric volume data interpolation
  std::vector<size_t> block_order{};

  // Integrate the ODE
  try {
    boost::numeric::odeint::integrate_times(
        stepper,
        [&](const Vars& state, Vars& dt_state, const double time) {
          // Wrap state and dt_state in tensors. This would be cleaner with a
          // Variables state type.
          const tnsr::I<double, Dim, Frame> position{
              {{state[0], state[1], state[2]}}};
          const tnsr::i<double, Dim, Frame> momentum{
              {{state[3], state[4], state[5]}}};
          const Scalar<double> redshift{state[6]};
          tnsr::I<double, Dim, Frame> dt_position{
              {{dt_state[0], dt_state[1], dt_state[2]}}};
          tnsr::i<double, Dim, Frame> dt_momentum{
              {{dt_state[3], dt_state[4], dt_state[5]}}};
          Scalar<double> dt_redshift{dt_state[6]};
          // Terminate ray near the horizon
          if (get(redshift) > 5.0) {
            throw std::runtime_error("Ray terminated near the horizon");
          }
          // Evaluate background at the position of the ray
          const auto background_vars = background_spacetime_->variables(
              position, time, make_not_null(&block_order));
          // Evaluate geodesic equation
          gr::geodesic_equation(
              make_not_null(&dt_position), make_not_null(&dt_momentum),
              make_not_null(&dt_redshift), position, momentum, redshift,
              get<gr::Tags::Lapse<double>>(background_vars),
              get<::Tags::deriv<gr::Tags::Lapse<double>, tmpl::size_t<Dim>,
                                Frame>>(background_vars),
              get<gr::Tags::Shift<double, Dim, Frame>>(background_vars),
              get<::Tags::deriv<gr::Tags::Shift<double, Dim, Frame>,
                                tmpl::size_t<Dim>, Frame>>(background_vars),
              get<gr::Tags::InverseSpatialMetric<double, Dim, Frame>>(
                  background_vars),
              get<::Tags::deriv<
                  gr::Tags::InverseSpatialMetric<double, Dim, Frame>,
                  tmpl::size_t<Dim>, Frame>>(background_vars),
              get<gr::Tags::ExtrinsicCurvature<double, Dim, Frame>>(
                  background_vars));
          // Unwrap dt_state back into the state array
          dt_state[0] = get<0>(dt_position);
          dt_state[1] = get<1>(dt_position);
          dt_state[2] = get<2>(dt_position);
          dt_state[3] = get<0>(dt_momentum);
          dt_state[4] = get<1>(dt_momentum);
          dt_state[5] = get<2>(dt_momentum);
          dt_state[6] = get(dt_redshift);
        },
        /* initial_state in result out */ ray, observer_times, initial_dt,
        std::ref(observer));
    // Ray integration finished successfully. Record the final state of the ray.
    if (verbosity_ >= ::Verbosity::Debug) {
      Parallel::printf("Done integrating ray %zu.\n", ray_index);
    }
    for (size_t j = 0; j < ray.size(); ++j) {
      ray_results_(i, j) = ray[j];
    }
  } catch (const std::exception& e) {
    // Ray integration stopped, possibly due to hitting a the horizon. Fill the
    // ray results with NaNs.
    if (verbosity_ >= ::Verbosity::Debug) {
      Parallel::printf("Stopped integrating ray %zu: %s\n", ray_index,
                       e.what());
    }
    for (size_t j = 0; j < ray.size(); ++j) {
      ray_results_(i, j) = std::numeric_limits<double>::signaling_NaN();
    }
  }
}

namespace {
std::pair<size_t, size_t> partition_ranks(const size_t total_num_rays,
                                          const size_t rank,
                                          const size_t num_ranks) {
  // Partition the rays evenly across ranks
  const size_t num_rays_per_rank = (total_num_rays + num_ranks - 1) / num_ranks;
  const size_t ray_index_start = rank * num_rays_per_rank;
  const size_t num_rays_this_rank =
      std::min(total_num_rays - ray_index_start, num_rays_per_rank);
  return {ray_index_start, num_rays_this_rank};
}
}  // namespace

void trace_frame(
    Kokkos::View<double**> ray_results, Kokkos::View<double***> ray_traces,
    const gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    const gsl::not_null<RaySource*> ray_source_ptr, const size_t frame,
    const size_t rank, const size_t num_ranks, const double abs_tol,
    const double rel_tol, const size_t num_output_steps,
    const std::optional<std::string>& output_file_name,
    const std::optional<std::string>& results_subfile_name,
    const std::optional<std::string>& traces_subfile_name,
    const ::Verbosity verbosity) {
  auto& background_spacetime = *background_spacetime_ptr;
  auto& ray_source = *ray_source_ptr;

  // Load data for this frame
  background_spacetime.initialize(ray_source.time_bounds(frame));

  // Initialize ray source given the background spacetime
  ray_source.initialize(frame, background_spacetime);

  // Get number of rays to trace for this rank
  const auto [ray_index_start, num_rays_this_rank] =
      partition_ranks(ray_source.num_rays(frame), rank, num_ranks);

  // Integrate geodesics in parallel
  if (verbosity >= ::Verbosity::Quiet) {
    Parallel::printf("Rank %zu: tracing %zu geodesics on %zu threads...\n",
                     rank, num_rays_this_rank, Kokkos::num_threads());
  }
  // Unwrap dynamic types because Kokkos and generally GPU code is not super
  // happy with polymorphism
  call_with_dynamic_type<void, all_background_spacetimes>(
      &background_spacetime,
      [&](const auto* const background_spacetime_derived) {
        call_with_dynamic_type<void, all_ray_sources>(
            &ray_source, [&](const auto* const ray_source_derived) {
              RayTracer ray_tracer(background_spacetime_derived,
                                   ray_source_derived, ray_results, ray_traces,
                                   ray_index_start, abs_tol, rel_tol,
                                   num_output_steps, verbosity);
              Kokkos::parallel_for("Trace geodesics", num_rays_this_rank,
                                   std::ref(ray_tracer));
              // Can also use OpenMP directly to enable dynamic scheduling
              // on CPUs:
              // #pragma omp parallel for schedule(dynamic)
              //   for (size_t i = 0; i < ray_index_start; ++i) {
              //     ray_tracer(i);
              //   }
            });
      });

  // Write results to disk if requested
  if (output_file_name.has_value()) {
    h5::H5File<h5::AccessType::ReadWrite> h5file(*output_file_name, true);
    std::vector<std::string> legend{"X",  "Y",  "Z",       "Px",
                                    "Py", "Pz", "Redshift"};
    std::vector<double> row_data(legend.size());
    // Write frame data
    if (results_subfile_name.has_value()) {
      auto& frame_datfile =
          h5file.insert<h5::Dat>(*results_subfile_name, legend);
      for (size_t i = 0; i < num_rays_this_rank; ++i) {
        for (size_t k = 0; k < legend.size(); ++k) {
          row_data[k] = ray_results(i, k);
        }  // for vars
        frame_datfile.append(row_data);
      }  // for num_rays_this_rank
      h5file.close_current_object();
    }
    // Write ray traces
    if (num_output_steps > 0 and traces_subfile_name.has_value()) {
      for (size_t i = 0; i < num_rays_this_rank; ++i) {
        auto& datfile = h5file.insert<h5::Dat>(
            *traces_subfile_name + "/Ray" + std::to_string(ray_index_start + i),
            legend);
        for (size_t j = 0; j < num_output_steps; ++j) {
          if (j > 0 and ray_traces(i, j, 6) == 0.0) {
            break;
          }
          for (size_t k = 0; k < legend.size(); ++k) {
            row_data[k] = ray_traces(i, j, k);
          }  // for vars
          datfile.append(row_data);
        }  // for num_output_steps
        h5file.close_current_object();
      }  // for num_rays
    }
  }  // if output_file_name
}

void trace_frames(
    const gsl::not_null<BackgroundSpacetime*> background_spacetime_ptr,
    const gsl::not_null<RaySource*> ray_source_ptr, const size_t rank,
    const size_t num_ranks, const double abs_tol, const double rel_tol,
    const size_t num_output_steps,
    const std::optional<std::string>& output_file_name,
    const ::Verbosity verbosity) {
  // Ray source keeps track of time
  auto& ray_source = *ray_source_ptr;
  // Allocate memory for output (will be resized later)
  const size_t num_vars = 7;
  Kokkos::View<double**> ray_results;
  Kokkos::View<double***> ray_traces;
  // Loop over frames
  const size_t num_frames = ray_source.num_frames();
  Kokkos::Timer timer{};
  for (size_t frame = 0; frame < num_frames; ++frame) {
    if (verbosity >= ::Verbosity::Quiet) {
      Parallel::printf("Rank %zu: tracing frame %zu...\n", rank, frame);
    }
    timer.reset();
    // Resize output arrays if needed
    const auto [ray_index_start, num_rays_this_rank] =
        partition_ranks(ray_source.num_rays(frame), rank, num_ranks);
    if (ray_results.extent(0) != num_rays_this_rank) {
      ray_results =
          Kokkos::View<double**>("RayResults", num_rays_this_rank, num_vars);
      ray_traces = Kokkos::View<double***>("RayTraces", num_rays_this_rank,
                                           num_output_steps, num_vars);
    }
    // Trace geodesics for this frame
    trace_frame(ray_results, ray_traces, background_spacetime_ptr,
                ray_source_ptr, frame, rank, num_ranks, abs_tol, rel_tol,
                num_output_steps, output_file_name,
                "Camera/Frame" + std::to_string(frame),
                "RayTraces/Frame" + std::to_string(frame));
    if (verbosity >= ::Verbosity::Quiet) {
      Parallel::printf("Rank %zu: frame %zu done in %g seconds.\n", rank, frame,
                       timer.seconds());
    }
  }
}

}  // namespace ray_tracing
