# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.TestHelpers.PointwiseFunctions.AnalyticSolutions.Xcts import (
    verify_constant_density_star)
from spectre.PointwiseFunctions.AnalyticSolutions.Xcts import (
    ConstantDensityStar)
import spectre.Domain.Creators as domain_creators
import unittest
import os


class TestVerifySolution(unittest.TestCase):
    def setUp(self):
        self.test_file_name = "TestVerifySolution.h5"
        if os.path.exists(self.test_file_name):
            os.remove(self.test_file_name)

    def tearDown(self):
        if os.path.exists(self.test_file_name):
            os.remove(self.test_file_name)

    def test_verify_constant_density_star(self):
        solution = ConstantDensityStar(density=0.01, radius=1.)
        domain = domain_creators.Brick(
            lower_xyz=[0., 0., 0.],
            upper_xyz=[2., 2., 2.],
            is_periodic_in_xyz=[False, False, False],
            initial_refinement_level_xyz=[1, 1, 1],
            initial_number_of_grid_points_in_xyz=[3, 3, 3])
        residuals = verify_constant_density_star(solution, domain,
                                                 self.test_file_name)
        self.assertEqual(len(residuals), 8)
        self.assertTrue(os.path.exists(self.test_file_name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
