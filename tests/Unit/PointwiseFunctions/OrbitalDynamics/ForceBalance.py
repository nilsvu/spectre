# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# def force_balance(eccentricity, position, conformal_factor_pow_4,
# lapse_square,
#             shift_square, shift_y, dx_conformal_factor_pow_4, dx_lapse_square,
#                dx_shift_square, dx_shift_y, center_of_mass, angular_velocity):
#     tangential_velocity = (1. - eccentricity) * \
#         angular_velocity * (position - center_of_mass)
#     shift_plus_vel_square = shift_square + 2. * shift_y * \
#         tangential_velocity + tangential_velocity**2
#     u_t_up_square = 1. / \
#         (lapse_square - conformal_factor_pow_4 * shift_plus_vel_square)
#     F = dx_lapse_square + \
#      dx_conformal_factor_pow_4 * (shift_square + tangential_velocity ** 2) + \
#         conformal_factor_pow_4 * (dx_shift_square + 2. * angular_velocity *
#                                   (shift_y + tangential_velocity)) - \
#    u_t_up_square * conformal_factor_pow_4 * shift_y * tangential_velocity * \
#         (dx_lapse_square - dx_conformal_factor_pow_4 * shift_plus_vel_square -
#conformal_factor_pow_4 *
#  (dx_shift_square + 2. * tangential_velocity * dx_shift_y))
#     return list(F)


def force_balance(eccentricity, position, conformal_factor_pow_4, lapse_square,
                  shift_square, shift_y, dx_conformal_factor_pow_4,
                  dx_lapse_square, dx_shift_square, dx_shift_y, center_of_mass,
                  angular_velocity):
    tangential_velocity = (1. - eccentricity) * \
        angular_velocity * (position - center_of_mass)
    shift_plus_vel_square = shift_square + 2. * shift_y * \
        tangential_velocity + tangential_velocity**2
    dx_shift_plus_vel_square = dx_shift_square + \
        2. * dx_shift_y * tangential_velocity
    u_t_up = 1. / \
        np.sqrt(lapse_square - conformal_factor_pow_4 * shift_plus_vel_square)
    dx_u_t_up = -0.5 * u_t_up ** 3 * (
        dx_lapse_square - dx_conformal_factor_pow_4 *
shift_plus_vel_square - conformal_factor_pow_4 * dx_shift_plus_vel_square)
  shift_square_plus_y_vel_square = shift_square + tangential_velocity * shift_y
    dx_shift_square_plus_y_vel_square = dx_shift_square + \
        tangential_velocity * dx_shift_y
    dx_u_t_lo = -dx_u_t_up * (lapse_square - conformal_factor_pow_4 * \
         shift_square_plus_y_vel_square) - \
        u_t_up * (dx_lapse_square - dx_conformal_factor_pow_4 * \
             shift_square_plus_y_vel_square -
                  conformal_factor_pow_4 * dx_shift_square_plus_y_vel_square)
    shift_plus_vel = shift_y + tangential_velocity
    dx_shift_plus_vel = dx_shift_y
    u_y_lo = u_t_up * conformal_factor_pow_4 * shift_plus_vel
    dx_u_y_lo = dx_u_t_up * conformal_factor_pow_4 * shift_plus_vel + u_t_up * \
         dx_conformal_factor_pow_4 * shift_plus_vel + \
        u_t_up * conformal_factor_pow_4 * dx_shift_plus_vel
    F = dx_u_t_lo + tangential_velocity * dx_u_y_lo + angular_velocity * u_y_lo
    return list(F)
