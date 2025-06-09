// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Factory.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "ParallelAlgorithms/RayTracer/RayTracer.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

struct RayTracerOptions : db::SimpleTag {
  static constexpr Options::String help =
      "Ray tracer that integrates geodesics in parallel";
  static std::string name() { return "RayTracer"; }

  using type = RayTracerOptions;
  using option_tags = tmpl::list<RayTracerOptions>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }

  struct AbsoluteTolerance {
    static constexpr Options::String help = "Absolute tolerance";
    using type = double;
  };
  struct RelativeTolerance {
    static constexpr Options::String help = "Relative tolerance";
    using type = double;
  };
  struct NumOutputSteps {
    static constexpr Options::String help = "Number of output steps";
    using type = size_t;
  };
  using options =
      tmpl::list<AbsoluteTolerance, RelativeTolerance, NumOutputSteps>;

  RayTracerOptions() = default;
  RayTracerOptions(const double abs_tol, const double rel_tol,
                   const size_t num_output_steps)
      : abs_tol(abs_tol),
        rel_tol(rel_tol),
        num_output_steps(num_output_steps) {}

  void pup(PUP::er& p) {
    p | abs_tol;
    p | rel_tol;
    p | num_output_steps;
  }

  double abs_tol;
  double rel_tol;
  size_t num_output_steps;
};

namespace Tags {
struct BackgroundSpacetime : db::SimpleTag {
  using type = std::unique_ptr<ray_tracing::BackgroundSpacetime>;
  static constexpr Options::String help =
      "Background spacetime for the ray tracer";
  using option_tags = tmpl::list<BackgroundSpacetime>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return value->get_clone();
  }
};
struct RaySource : db::SimpleTag {
  using type = std::unique_ptr<ray_tracing::RaySource>;
  static constexpr Options::String help =
      "Ray source for the ray tracer, which provides the initial conditions";
  using option_tags = tmpl::list<RaySource>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return value->get_clone();
  }
};
}  // namespace Tags

namespace Actions {

struct TraceGeodesics {
  using const_global_cache_tags =
      tmpl::list<RayTracerOptions, observers::Tags::VolumeFileName>;
  using simple_tags_from_options =
      tmpl::list<Tags::BackgroundSpacetime, Tags::RaySource>;
  using argument_tags = const_global_cache_tags;
  using return_tags = tmpl::list<Tags::BackgroundSpacetime, Tags::RaySource>;

  static void apply(const gsl::not_null<std::unique_ptr<BackgroundSpacetime>*>
                        background_spacetime,
                    const gsl::not_null<std::unique_ptr<RaySource>*> ray_source,
                    const RayTracerOptions& ray_tracer_options,
                    const std::string& output_file_name) {
    const size_t rank = static_cast<size_t>(sys::my_node());
    const size_t num_ranks = static_cast<size_t>(sys::number_of_nodes());
    trace_frames(background_spacetime->get(), ray_source->get(), rank,
                 num_ranks, ray_tracer_options.abs_tol,
                 ray_tracer_options.rel_tol,
                 ray_tracer_options.num_output_steps,
                 output_file_name + std::to_string(rank) + ".h5");
  }
};

}  // namespace Actions

template <class Metavariables>
struct RayTracerComponent {
  static constexpr size_t Dim = 3;

  using chare_type = Parallel::Algorithms::Nodegroup;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Execute,
          tmpl::list<::Actions::MutateApply<Actions::TraceGeodesics>,
                     Parallel::Actions::TerminatePhase>>>;

  using const_global_cache_tags = tmpl::list<>;

// using simple_tags_from_options = Parallel::get_simple_tags_from_options<
//     Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  using simple_tags_from_options =
      tmpl::list<Tags::BackgroundSpacetime, Tags::RaySource>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<RayTracerComponent>(local_cache)
        .start_phase(next_phase);
  }
};

struct Metavariables {
  static constexpr size_t Dim = 3;

  static constexpr Options::String help{
      "Trace geodesics through an analytic or numeric spacetime."};

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BackgroundSpacetime, all_background_spacetimes>,
                  tmpl::pair<RaySource, all_ray_sources>>;
  };

  using component_list = tmpl::list<RayTracerComponent<Metavariables>>;

  static constexpr auto default_phase_order =
      std::array{Parallel::Phase::Initialization, Parallel::Phase::Execute,
                 Parallel::Phase::Exit};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace ray_tracing
