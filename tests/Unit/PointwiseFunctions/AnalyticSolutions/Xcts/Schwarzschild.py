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
