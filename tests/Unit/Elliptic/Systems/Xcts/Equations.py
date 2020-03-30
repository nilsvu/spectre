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


def traceless_conformal_extrinsic_curvature(conformal_factor,
                                            lapse_times_conformal_factor,
                                            shift_strain):
    return (0.5 * conformal_factor**7 / lapse_times_conformal_factor *
            longitudinal_shift(shift_strain))


def traceless_conformal_extrinsic_curvature_square(
    conformal_factor, lapse_times_conformal_factor, shift_strain):
    return (conformal_factor**14 / lapse_times_conformal_factor**2 *
            (np.sum(shift_strain**2) - np.trace(shift_strain)**2 / 3.))


def hamiltonian_sources(energy_density, extrinsic_curvature_trace,
                        conformal_factor):
    return (conformal_factor**5 *
            (extrinsic_curvature_trace**2 / 12. - 2. * pi * energy_density))


def linearized_hamiltonian_sources(energy_density, extrinsic_curvature_trace,
                                   conformal_factor,
                                   conformal_factor_correction):
    return (5. * conformal_factor**4 * conformal_factor_correction *
            (extrinsic_curvature_trace**2 / 12. - 2. * pi * energy_density))


def lapse_sources(energy_density, stress_trace, extrinsic_curvature_trace,
                  conformal_factor, lapse_times_conformal_factor):
    return (lapse_times_conformal_factor * conformal_factor**4 *
            (5. / 12. * extrinsic_curvature_trace**2 + 2. * pi *
             (energy_density + 2. * stress_trace)))


def linearized_lapse_sources(energy_density, stress_trace,
                             extrinsic_curvature_trace, conformal_factor,
                             lapse_times_conformal_factor,
                             conformal_factor_correction,
                             lapse_times_conformal_factor_correction):
    return ((4. * lapse_times_conformal_factor * conformal_factor**3 *
             conformal_factor_correction +
             conformal_factor**4 * lapse_times_conformal_factor_correction) *
            (5. / 12. * extrinsic_curvature_trace**2 + 2. * pi *
             (energy_density + 2. * stress_trace)))


def momentum_sources(momentum_density, extrinsic_curvature_trace_gradient,
                     conformal_factor, lapse_times_conformal_factor, shift,
                     conformal_factor_gradient,
                     lapse_times_conformal_factor_gradient, shift_strain):
    return (np.einsum(
        'ij,j', longitudinal_shift(shift_strain),
        lapse_times_conformal_factor_gradient / lapse_times_conformal_factor -
        7. * conformal_factor_gradient / conformal_factor) +
            4. / 3. * lapse_times_conformal_factor / conformal_factor *
            extrinsic_curvature_trace_gradient +
            16. * pi * lapse_times_conformal_factor * conformal_factor**3 *
            momentum_density)


def linearized_momentum_sources(
    momentum_density, extrinsic_curvature_trace_gradient, conformal_factor,
    lapse_times_conformal_factor, shift, conformal_factor_gradient,
    lapse_times_conformal_factor_gradient, shift_strain,
    conformal_factor_correction, lapse_times_conformal_factor_correction,
    shift_correction, conformal_factor_gradient_correction,
    lapse_times_conformal_factor_gradient_correction, shift_strain_correction):
    return (np.einsum(
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
             7. * conformal_factor_gradient / conformal_factor) + 4. / 3. *
            (lapse_times_conformal_factor_correction / conformal_factor -
             lapse_times_conformal_factor / conformal_factor**2 *
             conformal_factor_correction) * extrinsic_curvature_trace_gradient
            + 16. * pi *
            (3. * lapse_times_conformal_factor * conformal_factor**2 *
             conformal_factor_correction + conformal_factor**3 *
             lapse_times_conformal_factor_correction) * momentum_density)


def shift_contribution_to_hamiltonian_sources(
    momentum_density, extrinsic_curvature_trace_gradient, conformal_factor,
    lapse_times_conformal_factor, shift, conformal_factor_gradient,
    lapse_times_conformal_factor_gradient, shift_strain):
    return (-1. / 8. * conformal_factor**7 / lapse_times_conformal_factor**2 *
            (np.sum(shift_strain**2) - np.trace(shift_strain)**2 / 3.))


def linearized_shift_contribution_to_hamiltonian_sources(
    momentum_density, extrinsic_curvature_trace_gradient, conformal_factor,
    lapse_times_conformal_factor, shift, conformal_factor_gradient,
    lapse_times_conformal_factor_gradient, shift_strain,
    conformal_factor_correction, lapse_times_conformal_factor_correction,
    shift_correction, conformal_factor_gradient_correction,
    lapse_times_conformal_factor_gradient_correction, shift_strain_correction):
    return -1. / 8. * (
        7. * conformal_factor**6 / lapse_times_conformal_factor**2 *
        (np.sum(shift_strain**2) - np.trace(shift_strain)**2 / 3.) *
        conformal_factor_correction -
        2. * conformal_factor**7 / lapse_times_conformal_factor**3 *
        (np.sum(shift_strain**2) - np.trace(shift_strain)**2 / 3.) *
        lapse_times_conformal_factor_correction +
        2. * conformal_factor**7 / lapse_times_conformal_factor**2 *
        (np.sum(shift_strain * shift_strain_correction) -
         np.trace(shift_strain) * np.trace(shift_strain_correction) / 3.))


def shift_contribution_to_lapse_sources(
    momentum_density, extrinsic_curvature_trace_gradient, conformal_factor,
    lapse_times_conformal_factor, shift, conformal_factor_gradient,
    lapse_times_conformal_factor_gradient, shift_strain):
    return (7. / 8. * conformal_factor**6 / lapse_times_conformal_factor *
            (np.sum(shift_strain**2) - np.trace(shift_strain)**2 / 3.) +
            conformal_factor**5 *
            np.sum(shift * extrinsic_curvature_trace_gradient))


def linearized_shift_contribution_to_lapse_sources(
    momentum_density, extrinsic_curvature_trace_gradient, conformal_factor,
    lapse_times_conformal_factor, shift, conformal_factor_gradient,
    lapse_times_conformal_factor_gradient, shift_strain,
    conformal_factor_correction, lapse_times_conformal_factor_correction,
    shift_correction, conformal_factor_gradient_correction,
    lapse_times_conformal_factor_gradient_correction, shift_strain_correction):
    return (
        7. / 8. *
        (6. * conformal_factor**5 / lapse_times_conformal_factor *
         (np.sum(shift_strain**2) - np.trace(shift_strain)**2 / 3.) *
         conformal_factor_correction -
         conformal_factor**6 / lapse_times_conformal_factor**2 *
         (np.sum(shift_strain**2) - np.trace(shift_strain)**2 / 3.) *
         lapse_times_conformal_factor_correction +
         2. * conformal_factor**6 / lapse_times_conformal_factor *
         (np.sum(shift_strain * shift_strain_correction) -
          np.trace(shift_strain) * np.trace(shift_strain_correction) / 3.)) +
        5. * conformal_factor**4 * conformal_factor_correction *
        np.sum(shift * extrinsic_curvature_trace_gradient) +
        conformal_factor**5 *
        np.sum(shift_correction * extrinsic_curvature_trace_gradient))
