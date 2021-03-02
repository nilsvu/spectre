# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import square, sqrt, abs


def normal_dot_conformal_factor_gradient(
    conformal_factor, lapse_times_conformal_factor,
    n_dot_longitudinal_shift_excess, x, extrinsic_curvature_trace,
    shift_background, longitudinal_shift_background, inv_conformal_metric,
    conformal_christoffel_second_kind):
    for i in range(3):
        for j in range(i):
            inv_conformal_metric[i, j] *= 1.e-3
            inv_conformal_metric[j, i] *= 1.e-3
        inv_conformal_metric[i, i] = abs(inv_conformal_metric[i, i])
    proper_radius = sqrt(np.einsum('ij,i,j', inv_conformal_metric, x, x))
    n = -x / proper_radius
    conformal_unit_normal = -n
    lapse = lapse_times_conformal_factor / conformal_factor
    n_dot_longitudinal_shift = n_dot_longitudinal_shift_excess + np.einsum(
        'i,ij', n, longitudinal_shift_background)
    # The following implements Eq. 7.12 in Harald's thesis
    conformal_unit_normal_raised = np.einsum('ij,j', inv_conformal_metric,
                                             conformal_unit_normal)
    inv_conformal_surface_metric = inv_conformal_metric - np.einsum(
        'i,j->ij', conformal_unit_normal_raised, conformal_unit_normal_raised)
    # Assuming here that the surface is a coordinate-sphere
    euclidean_radius = np.linalg.norm(x)
    unnormalized_face_normal = x / euclidean_radius
    magnitude_of_face_normal = sqrt(
        np.einsum('ij,i,j', inv_conformal_metric, unnormalized_face_normal,
                  unnormalized_face_normal)) # r_curved / r_flat
    deriv_unnormalized_face_normal = (
        np.identity(3) / euclidean_radius -
        np.einsum('i,j->ij', x, x) / euclidean_radius**3)
    # The term with the derivative of the magnitude vanishes when projected on
    # the surface metric, so we omit it here
    conformal_unit_normal_gradient = (
        deriv_unnormalized_face_normal / magnitude_of_face_normal - np.einsum(
            'i,ijk', conformal_unit_normal, conformal_christoffel_second_kind))
    projected_normal_gradient = np.einsum('ij,ij',
                                          inv_conformal_surface_metric,
                                          conformal_unit_normal_gradient)
    J = (2. / 3. * extrinsic_curvature_trace -
         0.5 / lapse * np.einsum('i,i', n, n_dot_longitudinal_shift))
    return (conformal_factor / 4. *
            (projected_normal_gradient - square(conformal_factor) * J))


def normal_dot_lapse_times_conformal_factor_gradient(
    conformal_factor, lapse_times_conformal_factor,
    n_dot_longitudinal_shift_excess, x, extrinsic_curvature_trace,
    shift_background, longitudinal_shift_background, inv_conformal_metric,
    conformal_christoffel_second_kind):
    return 0.


def linearized_normal_dot_lapse_times_conformal_factor_gradient(
    conformal_factor_correction, lapse_times_conformal_factor_correction,
    n_dot_longitudinal_shift_excess_correction, conformal_factor,
    lapse_times_conformal_factor, n_dot_longitudinal_shift_excess, x,
    extrinsic_curvature_trace, shift_background, longitudinal_shift_background,
    inv_conformal_metric, conformal_christoffel_second_kind):
    return 0.


def shift_excess(conformal_factor, lapse_times_conformal_factor,
                 n_dot_longitudinal_shift_excess, x, extrinsic_curvature_trace,
                 shift_background, longitudinal_shift_background,
                 inv_conformal_metric, conformal_christoffel_second_kind):
    for i in range(3):
        for j in range(i):
            inv_conformal_metric[i, j] *= 1.e-3
            inv_conformal_metric[j, i] *= 1.e-3
        inv_conformal_metric[i, i] = abs(inv_conformal_metric[i, i])
    proper_radius = sqrt(np.einsum('ij,i,j', inv_conformal_metric, x, x))
    n = -x / proper_radius
    conformal_unit_normal = -n
    lapse = lapse_times_conformal_factor / conformal_factor
    # The following implements Eq. 7.14 in Harald's thesis
    conformal_unit_normal_raised = np.einsum('ij,j', inv_conformal_metric,
                                             conformal_unit_normal)
    shift_orthogonal = lapse
    # TODO: add spin term
    shift_parallel = 0.
    return (shift_orthogonal * conformal_unit_normal_raised /
            square(conformal_factor) + shift_parallel - shift_background)
