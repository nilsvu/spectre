# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre import Spectral
from spectre import DataStructures
from spectre import Interpolation
import numpy as np

m1 = Spectral.Mesh1D(3, Spectral.Basis.Legendre, Spectral.Quadrature.Gauss)

m2 = Spectral.Mesh1D(5, Spectral.Basis.Legendre, Spectral.Quadrature.Gauss)

interpol = Interpolation.RegularGrid1D(m1, m2)

print(interpol.interpolate(DataStructures.DataVector(np.array([1., 2., 3.]))))
