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
