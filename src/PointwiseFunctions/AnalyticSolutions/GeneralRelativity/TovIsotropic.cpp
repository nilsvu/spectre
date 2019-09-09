// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TovIsotropic.hpp"

// Need Boost MultiArray because it is used internally by ODEINT
#include "DataStructures/BoostMultiArray.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <boost/numeric/odeint.hpp>  // IWYU pragma: keep
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>
#include <pup.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "ErrorHandling/Assert.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

#include "Parallel/Printf.hpp"

namespace gr {
namespace Solutions {

namespace {

template <typename F>
void dr_conformal_factor_potential(
    gsl::not_null<double*> dr_conformal_factor_potential,
    const double areal_radius, const F& mass_at_areal_radius) noexcept {
  const double sqrt_one_minus_two_m_over_r =
      sqrt(1. - 2. * mass_at_areal_radius(areal_radius) / areal_radius);
  *dr_conformal_factor_potential = (sqrt_one_minus_two_m_over_r - 1.) /
                                   (areal_radius * sqrt_one_minus_two_m_over_r);
}

class IntegralObserver {
 public:
  void operator()(const double current_conformal_factor_potential,
                  const double current_areal_radius) noexcept {
    areal_radius.push_back(current_areal_radius);
    conformal_factor_potential.push_back(current_conformal_factor_potential);
  }
  std::vector<double> areal_radius;
  std::vector<double> conformal_factor_potential;
};

template <typename DataType>
TovIsotropic::RadialVariables<DataType> interior_solution(
    const EquationsOfState::EquationOfState<true, 1>& /*equation_of_state*/,
    const DataType& radius, const DataType& conformal_factor,
    const DataType& log_specific_enthalpy,
    const double log_lapse_at_outer_radius) noexcept {
  TovIsotropic::RadialVariables<DataType> result{radius};
  result.conformal_factor = Scalar<DataType>{conformal_factor};
  result.specific_enthalpy = Scalar<DataType>{exp(log_specific_enthalpy)};
  //   result.rest_mass_density =
  //   equation_of_state.rest_mass_density_from_enthalpy(
  //       result.specific_enthalpy);
  //   result.pressure =
  //       equation_of_state.pressure_from_density(result.rest_mass_density);
  //   result.specific_internal_energy =
  //       equation_of_state.specific_internal_energy_from_density(
  //           result.rest_mass_density);
  //   result.metric_time_potential =
  //       log_lapse_at_outer_radius - log_specific_enthalpy;
  result.lapse =
      Scalar<DataType>{exp(log_lapse_at_outer_radius - log_specific_enthalpy)};
  //   result.dr_metric_time_potential =
  //       (mass_inside_radius + 4.0 * M_PI * get(result.pressure) *
  //       cube(radius)) / (radius * (radius - 2.0 * mass_inside_radius));
  //   result.metric_radial_potential =
  //       -0.5 * log(1.0 - 2.0 * mass_inside_radius / radius);
  //   result.dr_metric_radial_potential =
  //       (4.0 * M_PI * radius *
  //            (get(result.specific_enthalpy) * get(result.rest_mass_density) -
  //             get(result.pressure)) -
  //        mass_inside_radius / square(radius)) /
  //       (1.0 - 2.0 * mass_inside_radius / radius);
  //   result.metric_angular_potential = make_with_value<DataType>(radius, 0.0);
  //   result.dr_metric_angular_potential = make_with_value<DataType>(radius,
  //   0.0);
  return result;
}

template <typename DataType>
TovIsotropic::RadialVariables<DataType> vacuum_solution(
    const DataType& radius, const double total_mass) noexcept {
  TovIsotropic::RadialVariables<DataType> result{radius};
  const DataType m_over_two_r = 0.5 * total_mass / radius;
  const DataType one_plus_m_over_two_r = 1. + m_over_two_r;
  result.conformal_factor = Scalar<DataType>{one_plus_m_over_two_r};
  result.specific_enthalpy = make_with_value<Scalar<DataType>>(radius, 1.0);
  //   result.rest_mass_density =
  //   make_with_value<Scalar<DataType>>(radius, 0.0); result.pressure =
  //   make_with_value<Scalar<DataType>>(radius, 0.0);
  //   result.specific_internal_energy =
  //       make_with_value<Scalar<DataType>>(radius, 0.0);
  result.lapse = Scalar<DataType>{(1. - m_over_two_r) / one_plus_m_over_two_r};
  // result.dr_metric_time_potential =
  //     total_mass / square(radius) / one_minus_two_m_over_r;
  //   result.metric_radial_potential = -result.metric_time_potential;
  //   result.dr_metric_radial_potential =
  //   -result.dr_metric_time_potential; result.metric_angular_potential
  //   = make_with_value<DataType>(radius, 0.0);
  //   result.dr_metric_angular_potential =
  //   make_with_value<DataType>(radius, 0.0);
  return result;
}

}  // namespace

TovIsotropic::TovIsotropic(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const double central_mass_density,
    const double log_enthalpy_at_outer_radius, const double absolute_tolerance,
    const double relative_tolerance) {
  const TovSolution areal_solution{equation_of_state, central_mass_density,
                                   log_enthalpy_at_outer_radius,
                                   absolute_tolerance, relative_tolerance};
  const double areal_outer_radius = areal_solution.outer_radius();
  total_mass_ = areal_solution.mass(areal_outer_radius);
  log_lapse_at_outer_radius_ =
      0.5 * log(1. - 2. * total_mass_ / areal_outer_radius);
  // Parallel::printf("R=%f, M=%f\n", areal_outer_radius, total_mass_);
  const double outer_radius =
      0.5 * (areal_outer_radius - total_mass_ +
             sqrt(square(areal_outer_radius) -
                  2. * total_mass_ * areal_outer_radius));
  const auto mass_at_areal_radius = [&areal_solution](
      const double areal_radius) noexcept {
    return areal_solution.mass(areal_radius);
  };

  double current_areal_radius = 1.e-30 * areal_outer_radius;  // ?
  double current_conformal_factor_potential = 0.;             // ?
  double current_dr_conformal_factor_potential{};
  dr_conformal_factor_potential(&current_dr_conformal_factor_potential,
                                current_areal_radius, mass_at_areal_radius);

  const double initial_step = areal_outer_radius / 100.;  // ?
  using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<double>;
  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<StateDopri5>>
      dopri5 = make_dense_output(absolute_tolerance, relative_tolerance,
                                 StateDopri5{});
  IntegralObserver integral_observer{};
  boost::numeric::odeint::integrate_adaptive(
      dopri5,
      [&mass_at_areal_radius](const double /*local_conformal_factor_potential*/,
                              double& local_dr_conformal_factor_potential,
                              const double local_areal_radius) noexcept {
        return dr_conformal_factor_potential(
            &local_dr_conformal_factor_potential, local_areal_radius,
            mass_at_areal_radius);
      },
      current_conformal_factor_potential, current_areal_radius,
      areal_outer_radius, initial_step, std::ref(integral_observer));

  const double matching_constant =
      outer_radius / areal_outer_radius *
      exp(integral_observer.conformal_factor_potential.back());

  const size_t num_points = integral_observer.areal_radius.size();
  std::vector<double> isotropic_radius(num_points);
  std::vector<double> conformal_factor(num_points);
  std::vector<double> log_enthalpy(num_points);
  for (size_t i = 0; i < num_points; i++) {
    const double conformal_factor_square =
        exp(integral_observer.conformal_factor_potential[i]) /
        matching_constant;
    conformal_factor[i] = sqrt(conformal_factor_square);
    isotropic_radius[i] =
        integral_observer.areal_radius[i] / conformal_factor_square;
    log_enthalpy[i] =
        areal_solution.log_specific_enthalpy(integral_observer.areal_radius[i]);
  }
  // Parallel::printf("outer_radius=%f\n", matching_constant);
  // Parallel::printf("C=%f\n", matching_constant);
  // Parallel::printf("r=%s\n", integral_observer.areal_radius);
  // Parallel::printf("rbar=%s\n", isotropic_radius);
  // Parallel::printf("psi=%s\n", conformal_factor);

  // This should be equal to the outer_radius calculated before, but differs
  // by a numerical error
  outer_radius_ = isotropic_radius.back();
  conformal_factor_interpolant_ =
      intrp::CubicSpline(isotropic_radius, conformal_factor);
  log_enthalpy_interpolant_ =
      intrp::CubicSpline(isotropic_radius, log_enthalpy);
}

double TovIsotropic::outer_radius() const noexcept { return outer_radius_; }

template <>
TovIsotropic::RadialVariables<double> TovIsotropic::radial_variables(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const tnsr::I<double, 3>& x) const noexcept {
  // add small number to avoid FPEs at origin
  const double radius = get(magnitude(x)) + 1.e-30 * outer_radius_;
  if (radius >= outer_radius_) {
    return vacuum_solution(radius, total_mass_);
  }
  return interior_solution(equation_of_state, radius, conformal_factor(radius),
                           log_specific_enthalpy(radius),
                           log_lapse_at_outer_radius_);
}

template <>
TovIsotropic::RadialVariables<DataVector> TovIsotropic::radial_variables(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const tnsr::I<DataVector, 3>& x) const noexcept {
  // add small number to avoid FPEs at origin
  const DataVector radius = get(magnitude(x)) + 1.e-30 * outer_radius_;
  // Parallel::printf("r=%s\n", radius);
  if (min(radius) >= outer_radius_) {
    // Parallel::printf("all exterior...\n");
    return vacuum_solution(radius, total_mass_);
  }
  const size_t num_points = radius.size();
  if (max(radius) <= outer_radius_) {
    // Parallel::printf("all interior...\n");
    DataVector conformal_factor_at_radii{num_points};
    DataVector log_of_specific_enthalpy{num_points};
    for (size_t i = 0; i < num_points; i++) {
      const double r = radius[i];
      conformal_factor_at_radii[i] = conformal_factor(r);
      log_of_specific_enthalpy[i] = log_specific_enthalpy(r);
    }
    return interior_solution(
        equation_of_state, radius, conformal_factor_at_radii,
        log_of_specific_enthalpy, log_lapse_at_outer_radius_);
  }
  RadialVariables<DataVector> result{radius};
  // Parallel::printf("mixed...\n");
  for (size_t i = 0; i < num_points; i++) {
    const double r = radius[i];
    auto radial_vars_at_r =
        (r <= outer_radius_
             ? interior_solution(equation_of_state, r, conformal_factor(r),
                                 log_specific_enthalpy(r),
                                 log_lapse_at_outer_radius_)
             : vacuum_solution(r, total_mass_));
    get(result.conformal_factor)[i] = get(radial_vars_at_r.conformal_factor);
    // get(result.rest_mass_density)[i] =
    // get(radial_vars_at_r.rest_mass_density); get(result.pressure)[i] =
    // get(radial_vars_at_r.pressure); get(result.specific_internal_energy)[i] =
    //     get(radial_vars_at_r.specific_internal_energy);
    get(result.specific_enthalpy)[i] = get(radial_vars_at_r.specific_enthalpy);
    get(result.lapse)[i] = get(radial_vars_at_r.lapse);
    // result.dr_metric_time_potential[i] =
    //     radial_vars_at_r.dr_metric_time_potential;
    // result.metric_radial_potential[i] =
    //     radial_vars_at_r.metric_radial_potential;
    // result.dr_metric_radial_potential[i] =
    //     radial_vars_at_r.dr_metric_radial_potential;
  }
  //   result.metric_angular_potential = make_with_value<DataVector>(radius,
  //   0.0); result.dr_metric_angular_potential =
  //   make_with_value<DataVector>(radius, 0.0);
  return result;
}

void TovIsotropic::pup(PUP::er& p) noexcept {  // NOLINT
  p | outer_radius_;
  p | total_mass_;
  p | log_lapse_at_outer_radius_;
  p | conformal_factor_interpolant_;
  p | log_enthalpy_interpolant_;
}

}  // namespace Solutions
}  // namespace gr
