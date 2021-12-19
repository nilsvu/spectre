# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.elliptic.dg as elliptic_dg
import spectre.Domain.Creators as domain_creators
import spectre.DataStructures
import unittest
import numpy as np
import numpy.testing as npt


class TestXctsOperator(unittest.TestCase):
    def test_xcts_operator_matrix(self):
        domain_creator = domain_creators.Sphere(
            inner_radius=2,
            outer_radius=10,
            initial_refinement=0,
            initial_number_of_grid_points=[6, 6],
            use_equiangular_map=True)
        A_matrix_kerrschild = np.asarray(
            elliptic_dg.build_xcts_operator_matrix(domain_creator,
                                                   penalty_parameter=1.,
                                                   massive=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
