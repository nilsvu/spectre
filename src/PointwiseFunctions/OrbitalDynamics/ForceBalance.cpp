// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/OrbitalDynamics/ForceBalance.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
// #include "DataStructures/Tensor/Tensor.hpp"

#include "Parallel/Printf.hpp"

namespace orbital {

template <typename DataType>
DataType tangential_velocity(const double center_of_mass,
                             const double angular_velocity,
                             const DataType& position,
                             const double eccentricity) noexcept {
  return (1. - eccentricity) * angular_velocity * (position - center_of_mass);
}

// template <typename DataType>
// Scalar<DataType> lorentz_factor() noexcept {}

std::array<double, 2> ForceBalance::operator()(
    const std::array<double, 2>& center_of_mass_and_angular_velocity) const
    noexcept {
  const double center_of_mass = center_of_mass_and_angular_velocity[0];
  const double angular_velocity = center_of_mass_and_angular_velocity[1];

  const auto tangential_velocity = orbital::tangential_velocity(
      center_of_mass, angular_velocity, position, eccentricity);

  const auto shift_plus_velocity_square = shift_square +
                                          2. * shift_y * tangential_velocity +
                                          square(tangential_velocity);
  const auto dx_shift_plus_velocity_square =
      dx_shift_square + 2. * dx_shift_y * tangential_velocity;
  const auto u_t_up = 1. / sqrt(lapse_square - conformal_factor_pow_4 *
                                                   shift_plus_velocity_square);
  const auto dx_u_t_up =
      -0.5 * pow<3>(u_t_up) *
      (dx_lapse_square -
       dx_conformal_factor_pow_4 * shift_plus_velocity_square -
       conformal_factor_pow_4 * dx_shift_plus_velocity_square);
  const auto shift_square_plus_y_vel_square =
      shift_square + shift_y * tangential_velocity;
  const auto dx_shift_square_plus_y_vel_square =
      dx_shift_square + dx_shift_y * tangential_velocity;
  const auto dx_u_t_lo =
      -dx_u_t_up * (lapse_square -
                    conformal_factor_pow_4 * shift_square_plus_y_vel_square) -
      u_t_up * (dx_lapse_square -
                dx_conformal_factor_pow_4 * shift_square_plus_y_vel_square -
                conformal_factor_pow_4 * dx_shift_square_plus_y_vel_square);
  const auto shift_plus_vel = shift_y + tangential_velocity;
  const auto u_y_lo = u_t_up * conformal_factor_pow_4 * shift_plus_vel;
  const auto dx_u_y_lo = dx_u_t_up * conformal_factor_pow_4 * shift_plus_vel +
                         u_t_up * dx_conformal_factor_pow_4 * shift_plus_vel +
                         u_t_up * conformal_factor_pow_4 * dx_shift_y;
  //   const auto dx_u_x_lo = dx_u_t_up * conformal_factor_pow_4 * shift_x +
  //                          u_t_up * dx_conformal_factor_pow_4 * shift_x +
  //                          u_t_up * conformal_factor_pow_4 * dx_shift_x;

  const auto force_balance =
      dx_u_t_lo + tangential_velocity * dx_u_y_lo + angular_velocity * u_y_lo;

//   const auto u_t_up_square =
//    1. / (lapse_square - conformal_factor_pow_4 * shift_plus_velocity_square);
//   const auto force_balance =
//       dx_lapse_square +
//   dx_conformal_factor_pow_4 * (shift_square + square(tangential_velocity)) +
//       conformal_factor_pow_4 *
//           (dx_shift_square +
//            2. * angular_velocity * (shift_y + tangential_velocity)) -
//     u_t_up_square * conformal_factor_pow_4 * shift_y * tangential_velocity *
//           (dx_lapse_square -
//            dx_conformal_factor_pow_4 * shift_plus_velocity_square -
//            conformal_factor_pow_4 *
//                (dx_shift_square + 2. * tangential_velocity * dx_shift_y));

  Parallel::printf("Force balance for com=%e, w=%e: F = [%e, %e]\n",
                   center_of_mass, angular_velocity, force_balance[0],
                   force_balance[1]);
  return {{force_balance[0], force_balance[1]}};
}

std::array<std::array<double, 2>, 2> ForceBalance::jacobian(
    const std::array<double, 2>& center_of_mass_and_angular_velocity) const
    noexcept {
  const double center_of_mass = center_of_mass_and_angular_velocity[0];
  const double angular_velocity = center_of_mass_and_angular_velocity[1];

  const auto one_minus_ecc_times_dist_from_com =
      (1. - eccentricity) * (position - center_of_mass);
  const double dcom_one_minus_ecc_times_dist_from_com = eccentricity - 1.;
  const auto tangential_velocity =
      angular_velocity * one_minus_ecc_times_dist_from_com;
  const auto dcom_tangential_velocity =
      angular_velocity * dcom_one_minus_ecc_times_dist_from_com;
  const auto& dw_tangential_velocity = one_minus_ecc_times_dist_from_com;

  const auto shift_plus_velocity_square = shift_square +
                                          2. * shift_y * tangential_velocity +
                                          square(tangential_velocity);
  const auto dcom_shift_plus_velocity_square =
      2. * (shift_y + tangential_velocity) * dcom_tangential_velocity;
  const auto dw_shift_plus_velocity_square =
      2. * (shift_y + tangential_velocity) * dw_tangential_velocity;
  const auto dx_shift_plus_velocity_square =
      dx_shift_square + 2. * dx_shift_y * tangential_velocity;
  const auto dcom_dx_shift_plus_velocity_square =
      2. * dx_shift_y * dcom_tangential_velocity;
  const auto dw_dx_shift_plus_velocity_square =
      2. * dx_shift_y * dw_tangential_velocity;
  const auto u_t_up = 1. / sqrt(lapse_square - conformal_factor_pow_4 *
                                                   shift_plus_velocity_square);
  const auto dcom_u_t_up = 0.5 * pow<3>(u_t_up) * conformal_factor_pow_4 *
                           dcom_shift_plus_velocity_square;
  const auto dw_u_t_up = 0.5 * pow<3>(u_t_up) * conformal_factor_pow_4 *
                         dw_shift_plus_velocity_square;
  const auto dx_u_t_up =
      -0.5 * pow<3>(u_t_up) *
      (dx_lapse_square -
       dx_conformal_factor_pow_4 * shift_plus_velocity_square -
       conformal_factor_pow_4 * dx_shift_plus_velocity_square);
  const auto dcom_dx_u_t_up =
      -0.5 * 3. * square(u_t_up) * dcom_u_t_up *
          (dx_lapse_square -
           dx_conformal_factor_pow_4 * shift_plus_velocity_square -
           conformal_factor_pow_4 * dx_shift_plus_velocity_square) +
      0.5 * pow<3>(u_t_up) *
          (dx_conformal_factor_pow_4 * dcom_shift_plus_velocity_square +
           conformal_factor_pow_4 * dcom_dx_shift_plus_velocity_square);
  const auto dw_dx_u_t_up =
      -0.5 * 3. * square(u_t_up) * dw_u_t_up *
          (dx_lapse_square -
           dx_conformal_factor_pow_4 * shift_plus_velocity_square -
           conformal_factor_pow_4 * dx_shift_plus_velocity_square) +
      0.5 * pow<3>(u_t_up) *
          (dx_conformal_factor_pow_4 * dw_shift_plus_velocity_square +
           conformal_factor_pow_4 * dw_dx_shift_plus_velocity_square);
  const auto shift_square_plus_y_vel_square =
      shift_square + shift_y * tangential_velocity;
  const auto dcom_shift_square_plus_y_vel_square =
      shift_y * dcom_tangential_velocity;
  const auto dw_shift_square_plus_y_vel_square =
      shift_y * dw_tangential_velocity;
  const auto dx_shift_square_plus_y_vel_square =
      dx_shift_square + dx_shift_y * tangential_velocity;
  const auto dcom_dx_shift_square_plus_y_vel_square =
      dx_shift_y * dcom_tangential_velocity;
  const auto dw_dx_shift_square_plus_y_vel_square =
      dx_shift_y * dw_tangential_velocity;
  //   const auto dx_u_t_lo =
  //       -dx_u_t_up * (lapse_square -
  //                     conformal_factor_pow_4 *
  //                     shift_square_plus_y_vel_square) -
  //       u_t_up * (dx_lapse_square -
  //                 dx_conformal_factor_pow_4 * shift_square_plus_y_vel_square
  //                 - conformal_factor_pow_4 *
  //                 dx_shift_square_plus_y_vel_square);
  const auto dcom_dx_u_t_lo =
      -dcom_dx_u_t_up * (lapse_square - conformal_factor_pow_4 *
                                            shift_square_plus_y_vel_square) +
      dx_u_t_up * conformal_factor_pow_4 * dcom_shift_square_plus_y_vel_square -
      dcom_u_t_up *
          (dx_lapse_square -
           dx_conformal_factor_pow_4 * shift_square_plus_y_vel_square -
           conformal_factor_pow_4 * dx_shift_square_plus_y_vel_square) +
      u_t_up *
          (dx_conformal_factor_pow_4 * dcom_shift_square_plus_y_vel_square +
           conformal_factor_pow_4 * dcom_dx_shift_square_plus_y_vel_square);
  const auto dw_dx_u_t_lo =
      -dw_dx_u_t_up * (lapse_square - conformal_factor_pow_4 *
                                          shift_square_plus_y_vel_square) +
      dx_u_t_up * conformal_factor_pow_4 * dw_shift_square_plus_y_vel_square -
      dw_u_t_up * (dx_lapse_square -
                   dx_conformal_factor_pow_4 * shift_square_plus_y_vel_square -
                   conformal_factor_pow_4 * dx_shift_square_plus_y_vel_square) +
      u_t_up * (dx_conformal_factor_pow_4 * dw_shift_square_plus_y_vel_square +
                conformal_factor_pow_4 * dw_dx_shift_square_plus_y_vel_square);
  const auto shift_plus_vel = shift_y + tangential_velocity;
  const auto& dcom_shift_plus_vel = dcom_tangential_velocity;
  const auto& dw_shift_plus_vel = dw_tangential_velocity;
  const auto u_y_lo = u_t_up * conformal_factor_pow_4 * shift_plus_vel;
  const auto dcom_u_y_lo =
      dcom_u_t_up * conformal_factor_pow_4 * shift_plus_vel +
      u_t_up * conformal_factor_pow_4 * dcom_shift_plus_vel;
  const auto dw_u_y_lo = dw_u_t_up * conformal_factor_pow_4 * shift_plus_vel +
                         u_t_up * conformal_factor_pow_4 * dw_shift_plus_vel;
  const auto dx_u_y_lo = dx_u_t_up * conformal_factor_pow_4 * shift_plus_vel +
                         u_t_up * dx_conformal_factor_pow_4 * shift_plus_vel +
                         u_t_up * conformal_factor_pow_4 * dx_shift_y;
  const auto dcom_dx_u_y_lo =
      dcom_dx_u_t_up * conformal_factor_pow_4 * shift_plus_vel +
      dx_u_t_up * conformal_factor_pow_4 * dcom_shift_plus_vel +
      dcom_u_t_up * dx_conformal_factor_pow_4 * shift_plus_vel +
      u_t_up * dx_conformal_factor_pow_4 * dcom_shift_plus_vel +
      dcom_u_t_up * conformal_factor_pow_4 * dx_shift_y;
  const auto dw_dx_u_y_lo =
      dw_dx_u_t_up * conformal_factor_pow_4 * shift_plus_vel +
      dx_u_t_up * conformal_factor_pow_4 * dw_shift_plus_vel +
      dw_u_t_up * dx_conformal_factor_pow_4 * shift_plus_vel +
      u_t_up * dx_conformal_factor_pow_4 * dw_shift_plus_vel +
      dw_u_t_up * conformal_factor_pow_4 * dx_shift_y;

  const auto dcom_force_balance =
      dcom_dx_u_t_lo + dcom_tangential_velocity * dx_u_y_lo +
      tangential_velocity * dcom_dx_u_y_lo + angular_velocity * dcom_u_y_lo;
  const auto dw_force_balance = dw_dx_u_t_lo +
                                dw_tangential_velocity * dx_u_y_lo +
                                tangential_velocity * dw_dx_u_y_lo + u_y_lo +
                                angular_velocity * dw_u_y_lo;

//   const auto tangential_velocity = orbital::tangential_velocity(
//       center_of_mass, angular_velocity, position, eccentricity);
//const auto dcom_tangential_velocity = (eccentricity - 1.) * angular_velocity;
//   const auto dw_tangential_velocity =
//       (1. - eccentricity) * (position - center_of_mass);

//   const auto shift_plus_velocity = shift_y + tangential_velocity;
//   const auto shift_plus_velocity_square = shift_square +
//                                         2. * shift_y * tangential_velocity +
//                                           square(tangential_velocity);

//   const auto u_t_up_square =
//    1. / (lapse_square - conformal_factor_pow_4 * shift_plus_velocity_square);

//   const auto d_force_balance =
//       2. * tangential_velocity * dx_conformal_factor_pow_4 +
//       2. * conformal_factor_pow_4 * angular_velocity -
//       u_t_up_square * conformal_factor_pow_4 * shift_y *
//           (dx_lapse_square -
//            dx_conformal_factor_pow_4 * shift_plus_velocity_square -
//            conformal_factor_pow_4 *
//                (dx_shift_square + 2. * tangential_velocity * dx_shift_y)) *
//           (1. + 2. * u_t_up_square * conformal_factor_pow_4 *
//                     shift_plus_velocity * tangential_velocity) +
//       2. * u_t_up_square * conformal_factor_pow_4 * shift_y *
//           tangential_velocity *
//           (dx_conformal_factor_pow_4 * shift_plus_velocity +
//            conformal_factor_pow_4 * dx_shift_y);
//   const auto dcom_force_balance = dcom_tangential_velocity * d_force_balance;
//   const auto dw_force_balance =
//       2. * conformal_factor_pow_4 * shift_plus_velocity +
//       dw_tangential_velocity * d_force_balance;

  return {{{{dcom_force_balance[0], dw_force_balance[0]}},
           {{dcom_force_balance[1], dw_force_balance[1]}}}};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                   \
  template DTYPE(data) tangential_velocity(double, double, const DTYPE(data)&, \
                                           double) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace orbital
