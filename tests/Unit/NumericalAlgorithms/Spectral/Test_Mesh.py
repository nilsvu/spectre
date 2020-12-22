#Distributed under the MIT License.
#See LICENSE.txt for details.

from spectre.Spectral import Mesh1D, Mesh2D, Mesh3D
from spectre.Spectral import Basis, Quadrature

import unittest
import random
from math import prod


class TestMesh(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.bases = [Basis.Legendre, Basis.Chebyshev, Basis.FiniteDifference]
        self.quadratures = [
            Quadrature.Gauss, Quadrature.GaussLobatto, Quadrature.CellCentered,
            Quadrature.FaceCentered
        ]
        self.extents = range(12)

    def check_extents(self, mesh, extents):
        self.assertEqual(mesh.extents(0), extents[0])
        self.assertEqual(mesh.extents(), extents)
        self.assertEqual(mesh.number_of_grid_points(), prod(extents))

    def check_basis(self, mesh, bases):
        self.assertEqual(mesh.basis(0), bases[0])
        self.assertEqual(mesh.basis(), bases)

    def check_quadrature(self, mesh, quadratures):
        self.assertEqual(mesh.quadrature(0), quadratures[0])
        self.assertEqual(mesh.quadrature(), quadratures)

    def test_uniform(self):
        for extent in range(12):
            for basis in self.bases:
                for quadrature in self.quadratures:
                    m1 = Mesh1D(extent, basis, quadrature)
                    m2 = Mesh2D(extent, basis, quadrature)
                    m3 = Mesh3D(extent, basis, quadrature)

                    self.check_extents(m1, [extent])
                    self.check_extents(m2, [extent, extent])
                    self.check_extents(m3, [extent, extent, extent])

                    self.check_basis(m1, [basis])
                    self.check_basis(m2, [basis, basis])
                    self.check_basis(m3, [basis, basis, basis])

                    self.check_quadrature(m1, [quadrature])
                    self.check_quadrature(m2, [quadrature, quadrature])
                    self.check_quadrature(m3,
                                          [quadrature, quadrature, quadrature])

    def test_nonuniform_extents(self):
        for basis in self.bases:
            for quadrature in self.quadratures:
                for i in range(100):
                    extents1D = random.choices(self.extents, k=1)
                    extents2D = random.choices(self.extents, k=2)
                    extents3D = random.choices(self.extents, k=3)

                    m1 = Mesh1D(extents1D, basis, quadrature)
                    m2 = Mesh2D(extents2D, basis, quadrature)
                    m3 = Mesh3D(extents3D, basis, quadrature)

                    self.check_extents(m1, extents1D)
                    self.check_extents(m2, extents2D)
                    self.check_extents(m3, extents3D)

    def test_nonuniform_all(self):
        for i in range(100):
            extents1D = random.choices(self.extents, k=1)
            extents2D = random.choices(self.extents, k=2)
            extents3D = random.choices(self.extents, k=3)

            bases1D = random.choices(self.bases, k=1)
            bases2D = random.choices(self.bases, k=2)
            bases3D = random.choices(self.bases, k=3)

            quadratures1D = random.choices(self.quadratures, k=1)
            quadratures2D = random.choices(self.quadratures, k=2)
            quadratures3D = random.choices(self.quadratures, k=3)

            m1 = Mesh1D(extents1D, bases1D, quadratures1D)
            m2 = Mesh2D(extents2D, bases2D, quadratures2D)
            m3 = Mesh3D(extents3D, bases3D, quadratures3D)

            self.check_extents(m1, extents1D)
            self.check_extents(m2, extents2D)
            self.check_extents(m3, extents3D)

            self.check_basis(m1, bases1D)
            self.check_basis(m2, bases2D)
            self.check_basis(m3, bases3D)

            self.check_quadrature(m1, quadratures1D)
            self.check_quadrature(m2, quadratures2D)
            self.check_quadrature(m3, quadratures3D)


if __name__ == '__main__':
    unittest.main()
