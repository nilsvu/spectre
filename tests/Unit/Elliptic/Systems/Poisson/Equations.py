# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def euclidean_fluxes(field_gradient):
    return field_gradient


def noneuclidean_fluxes(det_metric, inv_metric, field_gradient):
    return np.sqrt(det_metric) * np.einsum('ij,j', inv_metric, field_gradient)


def auxiliary_fluxes(field, dim):
    return np.diag(np.repeat(field, dim))


def auxiliary_fluxes_1d(field):
    return auxiliary_fluxes(field, 1)


def auxiliary_fluxes_2d(field):
    return auxiliary_fluxes(field, 2)


def auxiliary_fluxes_3d(field):
    return auxiliary_fluxes(field, 3)
