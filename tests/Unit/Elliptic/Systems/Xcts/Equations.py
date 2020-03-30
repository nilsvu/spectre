# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import pi


def longitudinal_shift(shift_strain):
    result = 2. * shift_strain
    trace_term = 2. / 3. * np.trace(shift_strain)
    for d in range(3):
        result[d, d] -= trace_term
    return result


def hamiltonian_sources(energy_density, conformal_factor):
    return -2. * pi * conformal_factor**5 * energy_density


def linearized_hamiltonian_sources(energy_density, conformal_factor,
                                   conformal_factor_correction):
    return (-10. * pi * conformal_factor**4 * energy_density *
            conformal_factor_correction)


def lapse_sources(energy_density, stress_trace, conformal_factor,
                  lapse_times_conformal_factor):
    return 2. * pi * lapse_times_conformal_factor * conformal_factor**4 * (
        energy_density + 2. * stress_trace)


def linearized_lapse_sources(energy_density, stress_trace, conformal_factor,
                             lapse_times_conformal_factor,
                             conformal_factor_correction,
                             lapse_times_conformal_factor_correction):
    return 2. * pi * (4. * lapse_times_conformal_factor * conformal_factor**3 *
                      conformal_factor_correction + conformal_factor**4 *
                      lapse_times_conformal_factor_correction) * (
                          energy_density + 2. * stress_trace)


def momentum_sources(momentum_density, conformal_factor,
                     lapse_times_conformal_factor, conformal_factor_gradient,
                     lapse_times_conformal_factor_gradient, shift_strain):
    return np.einsum(
        'ij,j', longitudinal_shift(shift_strain),
        lapse_times_conformal_factor_gradient / lapse_times_conformal_factor -
        7. * conformal_factor_gradient / conformal_factor) + (
            16. * pi * lapse_times_conformal_factor * conformal_factor**3 *
            momentum_density)


def linearized_momentum_sources(
    momentum_density, conformal_factor, lapse_times_conformal_factor,
    conformal_factor_gradient, lapse_times_conformal_factor_gradient,
    shift_strain, conformal_factor_correction,
    lapse_times_conformal_factor_correction,
    conformal_factor_gradient_correction,
    lapse_times_conformal_factor_gradient_correction, shift_strain_correction):
    return np.einsum(
        'ij,j', longitudinal_shift(shift_strain),
        (lapse_times_conformal_factor_gradient_correction /
         lapse_times_conformal_factor - lapse_times_conformal_factor_gradient /
         (lapse_times_conformal_factor**2) *
         lapse_times_conformal_factor_correction) - 7. *
        (conformal_factor_gradient_correction / conformal_factor -
         conformal_factor_gradient / conformal_factor**2 *
         conformal_factor_correction)) + np.einsum(
             'ij,j', longitudinal_shift(shift_strain_correction),
             lapse_times_conformal_factor_gradient /
             lapse_times_conformal_factor -
             7. * conformal_factor_gradient / conformal_factor) + 16. * pi * (
                 3. * lapse_times_conformal_factor * conformal_factor**2 *
                 conformal_factor_correction + conformal_factor**3 *
                 lapse_times_conformal_factor_correction) * momentum_density


def shift_contribution_to_hamiltonian_sources(
    momentum_density, conformal_factor, lapse_times_conformal_factor,
    conformal_factor_gradient, lapse_times_conformal_factor_gradient,
    shift_strain):
    return -1. / 8. * conformal_factor**7 / lapse_times_conformal_factor**2 * (
        np.sum(shift_strain**2))


def linearized_shift_contribution_to_hamiltonian_sources(
    momentum_density, conformal_factor, lapse_times_conformal_factor,
    conformal_factor_gradient, lapse_times_conformal_factor_gradient,
    shift_strain, conformal_factor_correction,
    lapse_times_conformal_factor_correction,
    conformal_factor_gradient_correction,
    lapse_times_conformal_factor_gradient_correction, shift_strain_correction):
    return -1. / 8. * (
        7. * conformal_factor**6 / lapse_times_conformal_factor**2 *
        np.sum(shift_strain**2) * conformal_factor_correction -
        2. * conformal_factor**7 / lapse_times_conformal_factor**3 *
        np.sum(shift_strain**2) * lapse_times_conformal_factor_correction +
        2. * conformal_factor**7 / lapse_times_conformal_factor**2 *
        np.sum(shift_strain * shift_strain_correction))


def shift_contribution_to_lapse_sources(momentum_density, conformal_factor,
                                        lapse_times_conformal_factor,
                                        conformal_factor_gradient,
                                        lapse_times_conformal_factor_gradient,
                                        shift_strain):
    return 7. / 8. * conformal_factor**6 / lapse_times_conformal_factor * (
        np.sum(shift_strain**2))


def linearized_shift_contribution_to_lapse_sources(
    momentum_density, conformal_factor, lapse_times_conformal_factor,
    conformal_factor_gradient, lapse_times_conformal_factor_gradient,
    shift_strain, conformal_factor_correction,
    lapse_times_conformal_factor_correction,
    conformal_factor_gradient_correction,
    lapse_times_conformal_factor_gradient_correction, shift_strain_correction):
    return 7. / 8. * (
        6. * conformal_factor**5 / lapse_times_conformal_factor *
        np.sum(shift_strain**2) * conformal_factor_correction -
        conformal_factor**6 / lapse_times_conformal_factor**2 *
        np.sum(shift_strain**2) * lapse_times_conformal_factor_correction +
        conformal_factor**6 / lapse_times_conformal_factor *
        np.sum(shift_strain * shift_strain_correction))
