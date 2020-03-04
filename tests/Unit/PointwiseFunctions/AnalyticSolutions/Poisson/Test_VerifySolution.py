# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.TestHelpers.PointwiseFunctions.AnalyticSolutions.Poisson import (
    verify_product_of_sinusoids_1d)
from spectre.PointwiseFunctions.AnalyticSolutions.Poisson import (
    ProductOfSinusoids1D)
import spectre.Domain.Creators as domain_creators
from spectre.Domain import ElementId1D, SegmentId
import unittest
import numpy as np
import os


class TestVerifySolution(unittest.TestCase):
    def setUp(self):
        self.test_file_name = "TestVerifyPoissonSolution.h5"
        if os.path.exists(self.test_file_name):
            os.remove(self.test_file_name)

    def tearDown(self):
        if os.path.exists(self.test_file_name):
            os.remove(self.test_file_name)

    def test_verify_product_of_sinusoids(self):
        solution = ProductOfSinusoids1D(wave_numbers=[1])
        domain = domain_creators.Interval(
            lower_x=[0.],
            upper_x=[np.pi],
            is_periodic_in_x=[False],
            initial_refinement_level_x=[1],
            initial_number_of_grid_points_in_x=[3])
        residuals = verify_product_of_sinusoids_1d(solution, domain,
                                                   self.test_file_name)
        self.assertEqual(len(residuals), 2)
        left_element = ElementId1D(0, [SegmentId(1, 0)])
        right_element = ElementId1D(0, [SegmentId(1, 1)])
        self.assertAlmostEqual(residuals[left_element],
                               residuals[right_element])
        self.assertTrue(os.path.exists(self.test_file_name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
