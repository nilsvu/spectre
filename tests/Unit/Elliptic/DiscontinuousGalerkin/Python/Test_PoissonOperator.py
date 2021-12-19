# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.elliptic.dg as elliptic_dg
import spectre.Domain.Creators as domain_creators
import spectre.DataStructures
import unittest
import numpy as np
import numpy.testing as npt


class TestPoissonOperator(unittest.TestCase):
    def test_poisson_operator_matrix_1d(self):
        domain_creator = domain_creators.Interval(
            lower_x=[0],
            upper_x=[1],
            initial_refinement_level_x=[1],
            initial_number_of_grid_points_in_x=[3],
            is_periodic_in_x=[False])
        operator_matrix = np.asarray(
            elliptic_dg.build_poisson_operator_matrix_flat_cartesian_1d(
                domain_creator, penalty_parameter=1., massive=True))
        expected_operator_matrix = np.array(
            [[122., 8., -7., 3., 0., 0.], [8., 32., -4., -12., 0., 0.],
             [-7., -4., 68., -54., -12., 3.], [3., -12., -54., 68., -4., -7.],
             [0., 0., -12., -4., 32., 8.], [0., 0., 3., -7., 8., 122.]]) / 3.
        npt.assert_almost_equal(operator_matrix, expected_operator_matrix)


if __name__ == '__main__':
    unittest.main(verbosity=2)
