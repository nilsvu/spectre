// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolateAction.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/OrbitalDynamics/Tags.hpp"
#include "Utilities/TMPL.hpp"

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

namespace StarBinaryCenters_detail {

std::tuple<double, double, std::array<double, 2>> solve_force_balance(
    double eccentricity, const std::array<double, 2>& star_centers,
    const std::array<double, 2>& central_rest_mass_densities,
    const EquationsOfState::EquationOfState<true, 1>& eos,
    double center_of_mass_estimate, double angular_velocity_estimate,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& shift_gradient) noexcept;

}  // namespace StarBinaryCenters_detail

struct StarBinaryCenters {
  struct compute_target_points {
    template <typename ParallelComponent, typename DbTags,
              typename Metavariables, typename ArrayIndex,
              Requires<tmpl::list_contains_v<
                  DbTags, intrp::Tags::TemporalIds<Metavariables>>> = nullptr>
    static void apply(db::DataBox<DbTags>& box,
                      Parallel::ConstGlobalCache<Metavariables>& cache,
                      const ArrayIndex& /*array_index*/,
                      const size_t& temporal_id) noexcept {
      const auto& star_centers =
          get<typename Metavariables::initial_guess_tag>(cache).star_centers();
      intrp::send_points_to_interpolator<StarBinaryCenters>(
          box, cache,
          tnsr::I<DataVector, 3, Frame::Inertial>{
              {{{star_centers[0], star_centers[1]}, {0., 0.}, {0., 0.}}}},
          temporal_id);
    }
  };
  using compute_items_on_source = tmpl::list<>;
  using vars_to_interpolate_to_target = tmpl::list<
      Xcts::Tags::ConformalFactor<DataVector>,
      Xcts::Tags::LapseTimesConformalFactor<DataVector>,
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>>;
  using compute_items_on_target = tmpl::list<>;
  // Exposed for initialization
  using broadcast_tags =
      tmpl::list<orbital::Tags::CenterOfMass, orbital::Tags::AngularVelocity,
                 hydro::Tags::BinaryInjectionEnergy>;
  struct post_interpolation_callback {
    using OrbitalReductionData = Parallel::ReductionData<
        Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;
    using HydroReductionData = Parallel::ReductionData<
        Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;
    using observed_reduction_data_tags =
        observers::make_reduction_data_tags<
        tmpl::list<OrbitalReductionData, HydroReductionData>>;
    struct ObservationType {};
    using observation_types = tmpl::list<ObservationType>;
    template <typename DbTags, typename Metavariables>
    static void apply(const db::DataBox<DbTags>& box,
                      Parallel::ConstGlobalCache<Metavariables>& cache,
                      const size_t& temporal_id) noexcept {
      const auto& solution =
          get<typename Metavariables::initial_guess_tag>(cache);

      const auto force_balance_result =
          StarBinaryCenters_detail::solve_force_balance(
              solution.eccentricity(), solution.star_centers(),
              solution.central_rest_mass_densities(),
              solution.equation_of_state(), solution.center_of_mass_estimate(),
              solution.angular_velocity_estimate(),
              get<Xcts::Tags::ConformalFactor<DataVector>>(box),
              get<::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(box),
              get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(box),
              get<::Tags::deriv<
                  Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>>(box),
              get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(box),
              get<::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(box));
      const double center_of_mass = get<0>(force_balance_result);
      const double angular_velocity = get<1>(force_balance_result);
      const auto& binary_injection_energy = get<2>(force_balance_result);

      auto& array_component = Parallel::get_parallel_component<
          typename Metavariables::element_array_component>(cache);
      using temporal_id_tag = typename Metavariables::temporal_id;
      Parallel::receive_data<
          intrp::Tags::Broadcast<orbital::Tags::CenterOfMass, temporal_id_tag>>(
          array_component, temporal_id, center_of_mass);
      Parallel::receive_data<intrp::Tags::Broadcast<
          orbital::Tags::AngularVelocity, temporal_id_tag>>(
          array_component, temporal_id, angular_velocity);
      Parallel::receive_data<intrp::Tags::Broadcast<
          hydro::Tags::BinaryInjectionEnergy, temporal_id_tag>>(
          array_component, temporal_id, binary_injection_energy);

      const observers::ObservationId observation_id{
          static_cast<double>(temporal_id), ObservationType{}};
      auto& observer_writer_component = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);
      Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
          // Node 0 is always the writer, so directly call the component on that
          // node
          observer_writer_component[0], observation_id,
          std::string{"/orbital_dynamics"},
          std::vector<std::string>{db::tag_name<temporal_id_tag>(),
                                   "CenterOfMass", "AngularVelocity"},
          OrbitalReductionData{temporal_id, center_of_mass, angular_velocity});
      Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
          // Node 0 is always the writer, so directly call the component on that
          // node
          observer_writer_component[0], observation_id,
          std::string{"/hydro"},
          std::vector<std::string>{db::tag_name<temporal_id_tag>(),
                                   "LeftInjectionEnergy",
                                   "RightInjectionEnergy"},
          HydroReductionData{temporal_id, binary_injection_energy[0],
          binary_injection_energy[1]});
    }
  };
  using observed_reduction_data_tags =
      typename post_interpolation_callback::observed_reduction_data_tags;
};

}  // namespace InterpolationTargetTags
}  // namespace Xcts
