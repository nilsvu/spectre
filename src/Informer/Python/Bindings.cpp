// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
void bind_info_at_compile(py::module& m);
}  // namespace py_bindings

PYBIND11_MODULE(_PyInformer, m) {
  py_bindings::bind_info_at_compile(m);
}
