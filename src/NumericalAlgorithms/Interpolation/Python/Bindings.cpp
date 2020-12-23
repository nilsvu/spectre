// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
template <size_t Dim>
void bind_regular_grid(py::module& m);
}  // namespace py_bindings

PYBIND11_MODULE(_PyInterpolation, m) {  // NOLINT
  py_bindings::bind_regular_grid<1>(m);
  py_bindings::bind_regular_grid<2>(m);
  py_bindings::bind_regular_grid<3>(m);
}
