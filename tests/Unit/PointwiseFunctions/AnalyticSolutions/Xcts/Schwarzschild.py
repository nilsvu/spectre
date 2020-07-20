# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import sqrt, exp
from scipy.optimize import newton

# Isotropic Schwarzschild coordinates


def conformal_spatial_metric_isotropic(x):
    return np.identity(3)


def extrinsic_curvature_trace_isotropic(x):
    return 0.


def extrinsic_curvature_trace_gradient_isotropic(x):
    return np.zeros(3)


def conformal_factor_isotropic(x):
    r = np.linalg.norm(x)
    return 1. + 0.5 / r


def conformal_factor_gradient_isotropic(x):
    r = np.linalg.norm(x)
    return -0.5 * x / r**3


def lapse_times_conformal_factor_isotropic(x):
    r = np.linalg.norm(x)
    return 1. - 0.5 / r


def lapse_times_conformal_factor_gradient_isotropic(x):
    r = np.linalg.norm(x)
    return 0.5 * x / r**3


def shift_background(x):
    return np.zeros(3)


def shift_isotropic(x):
    return np.zeros(3)


def shift_strain_isotropic(x):
    return np.zeros((3, 3))


# Painleve-Gullstrand coordinates


def conformal_spatial_metric_painleve_gullstrand(x):
    return np.identity(3)


def extrinsic_curvature_trace_painleve_gullstrand(x):
    r = np.linalg.norm(x)
    return 1.5 * sqrt(2. / r**3)


def extrinsic_curvature_trace_gradient_painleve_gullstrand(x):
    r = np.linalg.norm(x)
    return -4.5 * sqrt(0.5 / r**5) * x / r


def conformal_factor_painleve_gullstrand(x):
    return 1.


def conformal_factor_gradient_painleve_gullstrand(x):
    return np.zeros(3)


def lapse_times_conformal_factor_painleve_gullstrand(x):
    return 1.


def lapse_times_conformal_factor_gradient_painleve_gullstrand(x):
    return np.zeros(3)


def shift_painleve_gullstrand(x):
    r = np.linalg.norm(x)
    return sqrt(2. / r) * x / r


def shift_strain_painleve_gullstrand(x):
    r = np.linalg.norm(x)
    return (sqrt(2. / r**3) *
            (np.identity(3) - 1.5 * np.tensordot(x, x, axes=0) / r**2))


# Areal Kerr-Schild coordinates

def lapse_kerr_schild(r):
    return 1. / sqrt(1. + 2. / r)


def lapse_kerr_schild_deriv(r):
    return lapse_kerr_schild(r)**3 / r**2


def shift_magnitude_kerr_schild(r):
    return 2. * lapse_kerr_schild(r)**2 / r


def shift_kerr_schild(x):
    r = np.linalg.norm(x)
    return shift_magnitude_kerr_schild(r) * x / r


def spatial_metric_kerr_schild(x):
    r = np.linalg.norm(x)
    return np.eye(3) + 2. * np.tensordot(x, x, axes=0) / r**3


def extrinsic_curvature_trace_kerr_schild(r):
    return 2. * lapse_kerr_schild(r)**3 / r**2 * (1. + 3. / r)


def extrinsic_curvature_trace_kerr_schild_deriv(r):
    lapse = lapse_kerr_schild(r)
    return (extrinsic_curvature_trace_kerr_schild(r) / r *
            (3. * lapse**2 / r - 3. / (r + 3.) - 2.))


def extrinsic_curvature_kerr_schild(x):
    r = np.linalg.norm(x)
    lapse = lapse_kerr_schild(r)
    return 2 * lapse / r**2 * (
        np.eye(3) - (2. + 1. / r) * np.tensordot(x, x, axes=0) / r**2)


# Isotropic Kerr-Schild coordinates

def kerr_schild_isotropic_radius_from_areal(r_areal):
    one_over_lapse = sqrt(1. + 2. / r_areal)
    return (r_areal / 4. * (1. + one_over_lapse)**2 *
            exp(2. - 2. * one_over_lapse))


def kerr_schild_isotropic_radius_from_areal_deriv(r_areal):
    one_over_lapse = sqrt(1. + 2. / r_areal)
    exp_term = exp(2. - 2. * one_over_lapse)
    return (0.25 * (1. + one_over_lapse)**2 - 0.5 *
            (1. + one_over_lapse) / r_areal / one_over_lapse + 0.5 *
            (1. + one_over_lapse)**2 / one_over_lapse / r_areal) * exp_term


def kerr_schild_areal_radius_from_isotropic(r_isotropic):
    def f(r_areal):
        return kerr_schild_isotropic_radius_from_areal(r_areal) - r_isotropic

    return newton(func=f,
                  fprime=kerr_schild_isotropic_radius_from_areal_deriv,
                  x0=r_isotropic)


def conformal_spatial_metric_kerr_schild_isotropic(x):
    return np.identity(3)


def extrinsic_curvature_trace_kerr_schild_isotropic(x):
    return extrinsic_curvature_trace_kerr_schild(
        kerr_schild_areal_radius_from_isotropic(np.linalg.norm(x)))


def extrinsic_curvature_trace_gradient_kerr_schild_isotropic(x):
    r_isotropic = np.linalg.norm(x)
    r_areal = kerr_schild_areal_radius_from_isotropic(r_isotropic)
    return (extrinsic_curvature_trace_kerr_schild_deriv(r_areal) /
            kerr_schild_isotropic_radius_from_areal_deriv(r_areal) * x /
            r_isotropic)


def conformal_factor_kerr_schild_isotropic_from_areal(r_areal):
    one_over_lapse = sqrt(1. + 2. / r_areal)
    return 2. * exp(one_over_lapse - 1.) / (1. + one_over_lapse)


def conformal_factor_kerr_schild_isotropic_from_areal_deriv(r_areal):
    one_over_lapse = sqrt(1. + 2. / r_areal)
    return (-2. * exp(one_over_lapse - 1.) / (1. + one_over_lapse)**2 /
            r_areal**2)


def conformal_factor_kerr_schild_isotropic(x):
    return conformal_factor_kerr_schild_isotropic_from_areal(
        kerr_schild_areal_radius_from_isotropic(np.linalg.norm(x)))


def conformal_factor_gradient_kerr_schild_isotropic(x):
    r_isotropic = np.linalg.norm(x)
    r_areal = kerr_schild_areal_radius_from_isotropic(r_isotropic)
    return (conformal_factor_kerr_schild_isotropic_from_areal_deriv(r_areal) /
            kerr_schild_isotropic_radius_from_areal_deriv(r_areal) * x /
            r_isotropic)


def lapse_times_conformal_factor_kerr_schild_isotropic(x):
    r_areal = kerr_schild_areal_radius_from_isotropic(np.linalg.norm(x))
    return (lapse_kerr_schild(r_areal) *
            conformal_factor_kerr_schild_isotropic_from_areal(r_areal))


def lapse_times_conformal_factor_gradient_kerr_schild_isotropic(x):
    r_isotropic = np.linalg.norm(x)
    r_areal = kerr_schild_areal_radius_from_isotropic(r_isotropic)
    return (
        (lapse_kerr_schild_deriv(r_areal) *
         conformal_factor_kerr_schild_isotropic_from_areal(r_areal) +
         lapse_kerr_schild(r_areal) *
         conformal_factor_kerr_schild_isotropic_from_areal_deriv(r_areal)) /
        kerr_schild_isotropic_radius_from_areal_deriv(r_areal) * x /
        r_isotropic)


def shift_magnitude_kerr_schild_isotropic_from_areal(r_areal):
    return (shift_magnitude_kerr_schild(r_areal) / lapse_kerr_schild(r_areal) /
            conformal_factor_kerr_schild_isotropic_from_areal(r_areal)**2)


def shift_magnitude_kerr_schild_isotropic_from_areal_deriv(r_areal):
    lapse = lapse_kerr_schild(r_areal)
    shift_magnitude = shift_magnitude_kerr_schild_isotropic_from_areal(r_areal)
    return (shift_magnitude / r_areal * ((lapse**2 + 2. * lapse /
                                          (1. + lapse)) / r_areal - 1.))


def shift_kerr_schild_isotropic(x):
    r_isotropic = np.linalg.norm(x)
    r_areal = kerr_schild_areal_radius_from_isotropic(r_isotropic)
    return (shift_magnitude_kerr_schild_isotropic_from_areal(r_areal) * x /
            r_isotropic)


def shift_strain_kerr_schild_isotropic(x):
    r_isotropic = np.linalg.norm(x)
    r_areal = kerr_schild_areal_radius_from_isotropic(r_isotropic)
    shift_magnitude = shift_magnitude_kerr_schild_isotropic_from_areal(r_areal)
    return ((shift_magnitude_kerr_schild_isotropic_from_areal_deriv(r_areal) /
             kerr_schild_isotropic_radius_from_areal_deriv(r_areal) -
             shift_magnitude / r_isotropic) * np.tensordot(x, x, axes=0) /
            r_isotropic**2 + shift_magnitude / r_isotropic * np.eye(3))


# Matter sources


def energy_density(x):
    return 0.


def stress_trace(x):
    return 0.


def momentum_density(x):
    return np.zeros(3)


# Fixed sources


def conformal_factor_fixed_source(x):
    return 0.


def lapse_times_conformal_factor_fixed_source(x):
    return 0.


def shift_fixed_source(x):
    return np.zeros(3)
