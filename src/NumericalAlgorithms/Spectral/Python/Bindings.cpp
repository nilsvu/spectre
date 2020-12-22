// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
void bind_basis(py::module& m);
void bind_quadrature(py::module& m);
template <size_t>
void bind_mesh(py::module& m);  // NOLINT

}  // namespace py_bindings

PYBIND11_MODULE(_PySpectral, m) {  // NOLINT
  py_bindings::bind_basis(m);
  py_bindings::bind_quadrature(m);
  py_bindings::bind_mesh<1>(m);
  py_bindings::bind_mesh<2>(m);
  py_bindings::bind_mesh<3>(m);
}
