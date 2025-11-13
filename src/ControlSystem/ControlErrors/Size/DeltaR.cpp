// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"

#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftInward.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"
#include "Utilities/StdHelpers.hpp"

namespace control_system::size::States {

std::unique_ptr<State> DeltaR::get_clone() const {
  return std::make_unique<DeltaR>(*this);
}

std::string DeltaR::update(const gsl::not_null<Info*> info,
                           const StateUpdateArgs& update_args,
                           const CrossingTimeInfo& crossing_time_info) const {
  // If update_args.control_error_delta_r is larger than
  // delta_r_control_error_threshold (and neither char speed nor
  // delta radius is in danger), then the timescale is decreased to
  // keep the control error small. This behavior is similar to what
  // TimecaleTuners do, but is triggered only in some situations. The
  // value of 1e-3 was chosen by trial and error in SpEC but it might
  // be helpful to decrease this value in the future if size control
  // needs to be very tight.
  constexpr double delta_r_control_error_threshold = 1.e-3;

  // Note that delta_radius_is_in_danger and char_speed_is_in_danger
  // can be different for different States.

  // The value of 0.99 was chosen by trial and error in SpEC.
  // It should be slightly less than unity but nothing should be
  // sensitive to small changes in this value.
  constexpr double time_tolerance_for_delta_r_in_danger = 0.99;
  const bool delta_radius_is_in_danger =
      (crossing_time_info.horizon_will_hit_excision_boundary_first and
       crossing_time_info.t_delta_radius_shrinking.value_or(
           std::numeric_limits<double>::infinity()) <
           info->damping_time * time_tolerance_for_delta_r_in_danger) or
      (crossing_time_info.horizon_is_expanding_too_fast and
       crossing_time_info.t_delta_radius_growing.value_or(
           std::numeric_limits<double>::infinity()) <
           info->damping_time * time_tolerance_for_delta_r_in_danger);
  const bool char_speed_is_in_danger =
      crossing_time_info.char_speed_will_hit_zero_first and
      crossing_time_info.t_char_speed.value_or(
          std::numeric_limits<double>::infinity()) < info->damping_time and
      not delta_radius_is_in_danger;

  // inward_drift_limit_in_danger means we are about
  // to cross the maximum allowed deltaR, and we are not in
  // the situation where we should exit state DeltaRNoDrift.
  // (the latter is important because we don't want to keep entering
  //  and exiting DeltaRNoDrift repeatedly)
  // In SpEC this variable is called "State2MaxLimitInDanger".
  const bool inward_drift_limit_in_danger =
      crossing_time_info.t_drift_limit_delta_radius.value_or(
          std::numeric_limits<double>::infinity()) < info->damping_time and
      update_args.min_allowed_radial_distance.has_value() and
      not ok_to_return_to_state_deltar(update_args);

  // This factor is present in SpEC, and it is used to prevent
  // oscillations between states.  The value was chosen in SpEC, but
  // nothing should be sensitive to small changes in this value as
  // long as it is slightly greater than unity.
  constexpr double non_oscillation_drift_outward_factor = 1.1;

  std::stringstream ss{};

  if (char_speed_is_in_danger) {
    ss << "Current state DeltaR. Char speed in danger.";
    if (crossing_time_info.t_comoving_char_speed.has_value() or
        update_args.min_comoving_char_speed < 0.0) {
      // Comoving char speed is negative or threatening to cross zero, so
      // staying in DeltaR mode will not work.  So switch to AhSpeed mode.

      // This factor prevents oscillating between states Initial and
      // AhSpeed.  It needs to be slightly greater than unity, but the
      // control system should not be sensitive to the exact
      // value. The value of 1.01 was chosen arbitrarily in SpEC and
      // never needed to be changed.
      constexpr double non_oscillation_factor = 1.01;
      info->discontinuous_change_has_occurred = true;
      info->state = std::make_unique<States::AhSpeed>();
      info->target_char_speed =
          update_args.min_char_speed * non_oscillation_factor;
      ss << " Switching to AhSpeed.\n";
      ss << " Target char speed = " << info->target_char_speed << "\n";
    } else {
      ss << " Staying in DeltaR.\n";
    }
    // If the comoving char speed is positive and is not about to
    // cross zero, staying in DeltaR mode will rescue the speed
    // automatically (since it drives char speed to comoving char
    // speed).  But we should decrease the timescale in any case.
    info->suggested_time_scale = crossing_time_info.t_char_speed;
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (delta_radius_is_in_danger) {
    if (crossing_time_info.horizon_will_hit_excision_boundary_first) {
      info->suggested_time_scale = crossing_time_info.t_delta_radius_shrinking;
    } else {
      info->suggested_time_scale = crossing_time_info.t_delta_radius_growing;
    }
    ss << "Current state DeltaR. Delta radius in danger. Staying in DeltaR.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (update_args.min_comoving_char_speed > 0.0 and
             std::abs(update_args.control_error_delta_r) >
                 delta_r_control_error_threshold) {
    // delta_r_state_decrease_factor should be slightly less than unity.
    // The value of 0.99 below was chosen arbitrarily in SpEC and never
    // needed to be changed.
    //
    // Note that this state is used in SpEC to reduce the timescale when the
    // control error grows too large, so the control system can react faster and
    // bring the control error down. This happens during plunge, for example,
    // when the horizon grows in the grid frame and the excision must follow
    // this growth.
    // However, there's a difference between SpEC and SpECTRE that changes how
    // often the `delta_r_state_decrease_factor` is applied: the _update
    // timescale_ is much smaller in SpEC than it is in SpECTRE, because in SpEC
    // it is tied to the timestep (an update happens every 3 or 4 timesteps, and
    // the timestep can decrease to lower the update timescale if needed)
    // whereas in SpECTRE the update timescale is a fixed fraction of the
    // damping time (see `Controller::UpdateFraction`). Therefore, the factor
    // `delta_r_state_decrease_factor` is applied much more frequently in SpEC
    // and so the timescale can decrease much faster in SpEC than in SpECTRE. We
    // want to keep the update timescale this large to avoid more frequent
    // horizon finds, but need to be able to reduce the timescale of size
    // control when the horizon is growing too quickly. Therefore, we added a
    // predictor similar to the zero-crossing predictor for a shrinking horizon
    // that estimates when the control error will exceed the threshold, and
    // uses that to set the timescale. This happens in the
    // `delta_radius_is_in_danger` block above. Therefore, this block here is
    // only reached if the control error is currently too large, but the horizon
    // is not predicted to cross the threshold within the damping time (unclear
    // if this can actually happen).
    //
    // For low spins it's not super important that the excision
    // follows the horizon very closely, but for high spins it's more important
    // because the excision should not cross into the inner Kerr horizon.
    // Therefore, for low spins it's probably enough to decrease the timescale
    // and try to keep DeltaR constant, whereas for high spins we may have to
    // switch to DeltaRDriftOutward (state 5) to decrease DeltaR.
    constexpr double delta_r_state_decrease_factor = 0.99;
    info->suggested_time_scale =
        info->damping_time * delta_r_state_decrease_factor;
    ss << "Current state DeltaR. Min comoving char speed "
       << update_args.min_comoving_char_speed
       << " > 0 and abs(control_error_delta_r) "
       << std::abs(update_args.control_error_delta_r) << " > threshold "
       << delta_r_control_error_threshold << ". Staying in DeltaR.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (should_transition_from_state_delta_r_to_inward_drift(
                 crossing_time_info.t_drift_limit, info->damping_time,
                 update_args)) {
    info->discontinuous_change_has_occurred = true;
    info->state = std::make_unique<States::DeltaRDriftInward>();
    info->target_char_speed = target_speed_for_inward_drift(
        update_args.avg_distorted_normal_dot_unit_coord_vector,
        update_args.min_char_speed, update_args.inward_drift_velocity.value());
    ss << "Current state DeltaR. "
          "Horizon too close to excision boundary. Switching to "
          "DeltaRDriftInward";
  } else if (inward_drift_limit_in_danger) {
    info->suggested_time_scale = crossing_time_info.t_drift_limit_delta_radius;
    ss << "Current state DeltaR. Inward drift limit in danger. Staying "
          "in DeltaR.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (update_args.average_radial_distance.has_value() and
             update_args.average_radial_distance.value() >
                 non_oscillation_drift_outward_factor *
                     update_args.max_allowed_radial_distance.value_or(
                         std::numeric_limits<double>::infinity())) {
    info->discontinuous_change_has_occurred = true;
    info->state = std::make_unique<States::DeltaRDriftOutward>();
    ss << "Current state DeltaR. "
          "Horizon too far from excision boundary. Switching to "
          "DeltaRDriftOutward";
  } else {
    ss << "Current state DeltaR. No change necessary. Staying in DeltaR.";
  }

  return ss.str();
}

double DeltaR::control_error(const Info& /*info*/,
                             const ControlErrorArgs& control_error_args) const {
  return control_error_args.control_error_delta_r;
}

PUP::able::PUP_ID DeltaR::my_PUP_ID = 0;
}  // namespace control_system::size::States
