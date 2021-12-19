// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace elliptic::dg {

namespace py_bindings {
void bind_poisson_operator(py::module& m);     // NOLINT
void bind_elasticity_operator(py::module& m);  // NOLINT
void bind_xcts_operator(py::module& m);        // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyEllipticDg, m) {  // NOLINT
  py_bindings::bind_poisson_operator(m);
  py_bindings::bind_elasticity_operator(m);
  py_bindings::bind_xcts_operator(m);
}

}  // namespace elliptic::dg
