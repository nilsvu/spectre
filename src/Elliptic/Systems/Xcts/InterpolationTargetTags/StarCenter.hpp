// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolateAction.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"

#include "Parallel/Printf.hpp"

/// \cond
namespace observers {
namespace ThreadedActions {
struct WriteReductionData;
}  // namespace ThreadedActions
template <class Metavariables>
struct ObserverWriter;
}  // namespace observers
/// \endcond

namespace Xcts {
namespace InterpolationTargetTags {

struct StarCenter {
  struct compute_target_points {
    template <typename ParallelComponent, typename DbTags,
              typename Metavariables, typename ArrayIndex,
              Requires<tmpl::list_contains_v<
                  DbTags, intrp::Tags::TemporalIds<Metavariables>>> = nullptr>
    static void apply(db::DataBox<DbTags>& box,
                      Parallel::ConstGlobalCache<Metavariables>& cache,
                      const ArrayIndex& /*array_index*/,
                      const size_t& temporal_id) noexcept {
      intrp::send_points_to_interpolator<StarCenter>(
          box, cache,
          tnsr::I<DataVector, 3, Frame::Inertial>{{{{0.}, {0.}, {0.}}}},
          temporal_id);
    }
  };
  using compute_items_on_source = tmpl::list<>;
  using vars_to_interpolate_to_target =
      tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactor<DataVector>>;
  using compute_items_on_target = tmpl::list<>;
  // Exposed for initialization
  using broadcast_tags = tmpl::list<hydro::Tags::InjectionEnergy>;
  struct post_interpolation_callback {
    using ReductionData = Parallel::ReductionData<
        Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;
    using observed_reduction_data_tags =
        observers::make_reduction_data_tags<tmpl::list<ReductionData>>;
    struct ObservationType {};
    using observation_types = tmpl::list<ObservationType>;
    template <typename DbTags, typename Metavariables>
    static void apply(const db::DataBox<DbTags>& box,
                      Parallel::ConstGlobalCache<Metavariables>& cache,
                      const size_t& temporal_id) noexcept {
      const double lapse_at_center =
          get(get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(box))[0] /
          get(get<Xcts::Tags::ConformalFactor<DataVector>>(box))[0];
      const auto& solution =
          get<typename Metavariables::initial_guess_tag>(cache);
      const auto& eos = solution.equation_of_state();
      const double specific_enthalpy_at_center =
          get(eos.specific_enthalpy_from_density(
              Scalar<double>(solution.central_rest_mass_density())));
      const double injection_energy =
          specific_enthalpy_at_center * lapse_at_center;

      Parallel::printf("lapse at center: %e\n", lapse_at_center);
      Parallel::printf("injection energy: %e\n", injection_energy);

      using temporal_id_tag = typename Metavariables::temporal_id;
      Parallel::receive_data<intrp::Tags::Broadcast<
          hydro::Tags::InjectionEnergy, temporal_id_tag>>(
          Parallel::get_parallel_component<
              typename Metavariables::element_array_component>(cache),
          temporal_id, injection_energy);

      const observers::ObservationId observation_id{
          static_cast<double>(temporal_id), ObservationType{}};
      auto& observer_writer_component = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);
      Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
          // Node 0 is always the writer, so directly call the component on that
          // node
          observer_writer_component[0], observation_id,
          std::string{"/hydro"},
          std::vector<std::string>{db::tag_name<temporal_id_tag>(),
                                   "InjectionEnergy"},
          ReductionData{temporal_id, injection_energy});
    }
  };
  using observed_reduction_data_tags =
      typename post_interpolation_callback::observed_reduction_data_tags;
};

}  // namespace InterpolationTargetTags
}  // namespace Xcts
