# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from scipy.optimize import newton


def extrinsic_curvature_trace(x, density, radius):
    return 0.


def longitudinal_shift_minus_dt_conformal_metric_over_lapse_square(
    x, density, radius):
    return 0.


def energy_density(x, density, radius):
    if np.linalg.norm(x) <= radius:
        return density
    else:
        return 0.


def compute_alpha(density, radius):
    def f(a):
        return density * radius**2 - 3. / (2. * np.pi) * a**10 / (1. + a**2)**6

    def fprime(a):
        return 3. * a**9 * (a**2 - 5.) / (1. + a**2)**7 / np.pi

    return newton(func=f, fprime=fprime, x0=2. * np.sqrt(5.))


def compute_inner_prefactor(density):
    return (3. / (2. * np.pi * density))**(1. / 4.)


def sobolov(r, alpha, radius):
    return np.sqrt(alpha * radius / (r**2 + (alpha * radius)**2))


def sobolov_dr_over_r(r, alpha, radius):
    return -np.sqrt(alpha * radius / (r**2 + (alpha * radius)**2)**3)


def conformal_factor(x, density, radius):
    alpha = compute_alpha(density, radius)
    C = compute_inner_prefactor(density)
    r = np.linalg.norm(x)
    if r <= radius:
        return C * sobolov(r, alpha, radius)
    else:
        beta = radius * (C * sobolov(radius, alpha, radius) - 1.)
        return beta / r + 1.


def conformal_factor_gradient(x, density, radius):
    alpha = compute_alpha(density, radius)
    C = compute_inner_prefactor(density)
    r = np.linalg.norm(x)
    if r <= radius:
        return -C * sobolov_dr_over_r(r, alpha, radius) * x
    else:
        beta = radius * (C * sobolov(radius, alpha, radius) - 1.)
        return -beta / r**3 * x


def initial_conformal_factor(x, density, radius):
    return 1.


def initial_conformal_factor_gradient(x, density, radius):
    return np.zeros(len(x))


def conformal_factor_source(x, density, radius):
    return 0.
