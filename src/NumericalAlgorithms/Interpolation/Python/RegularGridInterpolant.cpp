// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace py_bindings {

template <size_t Dim>
void bind_regular_grid(py::module& m) {  // NOLINT
  // the bindings are not complete and can be extended
  py::class_<intrp::RegularGrid<Dim>>(
      m, ("RegularGrid" + std::to_string(Dim) + "D").c_str())
      .def(py::init([](const Mesh<Dim>& m1, const Mesh<Dim>& m2) {
        return intrp::RegularGrid<Dim>(m1, m2);
      }))
      .def("interpolate",
           static_cast<DataVector (intrp::RegularGrid<Dim>::*)(
               const DataVector&) const>(&intrp::RegularGrid<Dim>::interpolate))
      .def("interpolation_matrices",
           &intrp::RegularGrid<Dim>::interpolation_matrices);
}
template void bind_regular_grid<1>(py::module& m);
template void bind_regular_grid<2>(py::module& m);
template void bind_regular_grid<3>(py::module& m);

}  // namespace py_bindings
