# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.elliptic.dg as elliptic_dg
import spectre.Domain.Creators as domain_creators
import spectre.DataStructures
import unittest
import numpy as np
import numpy.testing as npt


class TestElasticityOperator(unittest.TestCase):
    def test_poisson_operator_matrix_1d(self):
        domain_creator = domain_creators.Rectangle(
            lower_xy=[0., 0.],
            upper_xy=[1., 1.],
            initial_refinement_level_xy=[1, 1],
            initial_number_of_grid_points_in_xy=[3, 3],
            is_periodic_in_xy=[False, False])
        operator_matrix = np.asarray(
            elliptic_dg.build_elasticity_operator_matrix_2d(
                domain_creator, penalty_parameter=1., massive=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
