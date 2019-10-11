// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/InterpolationTargetTags/StarBinaryCenters.hpp"

#include <array>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/GslMultiRoot.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/OrbitalDynamics/ForceBalance.hpp"
#include "Utilities/ConstantExpressions.hpp"

#include "Parallel/Printf.hpp"

namespace Xcts {
namespace InterpolationTargetTags {
namespace StarBinaryCenters_detail {

std::tuple<double, double, std::array<double, 2>> solve_force_balance(
    const double eccentricity, const std::array<double, 2>& star_centers,
    const std::array<double, 2>& central_rest_mass_densities,
    const EquationsOfState::EquationOfState<true, 1>& eos,
    const double center_of_mass_estimate,
    const double angular_velocity_estimate,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& shift_gradient) noexcept {
  const DataVector local_star_centers{star_centers[0], star_centers[1]};
  const auto conformal_factor_pow_4 = pow<4>(get(conformal_factor));
  const auto lapse = get(lapse_times_conformal_factor) / get(conformal_factor);
  const auto lapse_square = square(lapse);
  const auto shift_square = get(dot_product(shift, shift));
  const auto& shift_y = get<1>(shift);
  const auto& dx_conformal_factor = get<0>(conformal_factor_gradient);
  const auto dx_conformal_factor_pow_4 =
      4. * pow<3>(get(conformal_factor)) * dx_conformal_factor;
  const auto& dx_lapse_times_conformal_factor =
      get<0>(lapse_times_conformal_factor_gradient);
  const auto dx_lapse =
      (dx_lapse_times_conformal_factor - lapse * dx_conformal_factor) /
      get(conformal_factor);
  const auto dx_lapse_square = 2. * lapse * dx_lapse;
  const auto dx_shift_square = 2. * (get<0>(shift) * get<0, 0>(shift_gradient) +
                                     get<1>(shift) * get<0, 1>(shift_gradient) +
                                     get<2>(shift) * get<0, 2>(shift_gradient));
  const auto& dx_shift_y = get<0, 1>(shift_gradient);

  Parallel::printf("eccentricity = %e\n", eccentricity);
  Parallel::printf("position = [%e, %e]\n", local_star_centers[0],
                   local_star_centers[1]);
  Parallel::printf("conformal_factor_pow_4 = [%e, %e]\n",
                   conformal_factor_pow_4[0], conformal_factor_pow_4[1]);
  Parallel::printf("dx_conformal_factor_pow_4 = [%e, %e]\n",
                   dx_conformal_factor_pow_4[0], dx_conformal_factor_pow_4[1]);
  Parallel::printf("lapse_square = [%e, %e]\n",
  lapse_square[0], lapse_square[1]);
  Parallel::printf("dx_lapse_square = [%e, %e]\n", dx_lapse_square[0],
                   dx_lapse_square[1]);
  Parallel::printf("shift_square = [%e, %e]\n", shift_square[0],
  shift_square[1]);
  Parallel::printf("dx_shift_square = [%e, %e]\n", dx_shift_square[0],
                   dx_shift_square[1]);
  Parallel::printf("shift_y = [%e, %e]\n", shift_y[0], shift_y[1]);
  Parallel::printf("dx_shift_y = [%e, %e]\n", dx_shift_y[0], dx_shift_y[1]);

  Parallel::printf("initial guess for angular_velocity: %e\n",
                   angular_velocity_estimate);
  std::array<double, 2> initial_center_of_mass_and_angular_velocity{
      {center_of_mass_estimate, angular_velocity_estimate}};
  const orbital::ForceBalance force_balance{eccentricity,
                                            local_star_centers,
                                            conformal_factor_pow_4,
                                            lapse_square,
                                            shift_square,
                                            shift_y,
                                            dx_conformal_factor_pow_4,
                                            dx_lapse_square,
                                            dx_shift_square,
                                            dx_shift_y};
//   const auto center_of_mass_and_angular_velocity = RootFinder::gsl_multiroot(
//    force_balance, initial_center_of_mass_and_angular_velocity, 1.e-14, 20);
  const auto center_of_mass_and_angular_velocity =
      initial_center_of_mass_and_angular_velocity;
  force_balance(center_of_mass_and_angular_velocity);

  const DataVector specific_enthalpy = get(eos.specific_enthalpy_from_density(
      Scalar<DataVector>{{{{central_rest_mass_densities[0],
                            central_rest_mass_densities[1]}}}}));

  const double center_of_mass = center_of_mass_and_angular_velocity[0];
  const double angular_velocity = center_of_mass_and_angular_velocity[1];
  const auto tangential_velocity = orbital::tangential_velocity(
      center_of_mass, angular_velocity, local_star_centers, eccentricity);

  Parallel::printf("center_of_mass = %e\n", center_of_mass);
  Parallel::printf("angular_velocity = %e\n", angular_velocity);
  Parallel::printf("tangential_velocity = %s\n", tangential_velocity);

  // Move this elsewhere
  const auto injection_energy =
      specific_enthalpy *
      sqrt(lapse_square -
           conformal_factor_pow_4 *
               (shift_square + 2. * tangential_velocity * shift_y +
                square(tangential_velocity)));

  Parallel::printf("injection_energy = [%e, %e]\n", injection_energy[0],
                   injection_energy[1]);

  return {center_of_mass,
          angular_velocity,
          {{injection_energy[0], injection_energy[1]}}};
}

}  // namespace StarBinaryCenters_detail
}  // namespace InterpolationTargetTags
}  // namespace Xcts
