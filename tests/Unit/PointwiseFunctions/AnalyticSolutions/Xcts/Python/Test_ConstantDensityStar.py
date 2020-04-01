# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.PointwiseFunctions.AnalyticSolutions.Xcts import (
    ConstantDensityStar)
import unittest


class TestConstantDensityStar(unittest.TestCase):
    def test_construction(self):
        solution = ConstantDensityStar(density=0.01, radius=1.)
        self.assertEqual(solution.density(), 0.01)
        self.assertEqual(solution.radius(), 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
