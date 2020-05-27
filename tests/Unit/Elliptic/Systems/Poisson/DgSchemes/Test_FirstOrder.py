# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.TestHelpers.Poisson.DgSchemes import (
    first_order_operator_matrix_1d)
import spectre.Domain.Creators as domain_creators
import spectre.DataStructures
import unittest
import numpy as np
import numpy.testing as npt


class TestFirstOrderDgScheme(unittest.TestCase):
    def test_operator_matrix(self):
        domain = domain_creators.Interval(
            lower_x=[0.],
            upper_x=[1.],
            is_periodic_in_x=[False],
            initial_refinement_level_x=[1],
            initial_number_of_grid_points_in_x=[3])
        operator_matrix = first_order_operator_matrix_1d(domain, 6.75)
        self.assertEqual(operator_matrix.shape, (12, 12))


if __name__ == '__main__':
    unittest.main(verbosity=2)
