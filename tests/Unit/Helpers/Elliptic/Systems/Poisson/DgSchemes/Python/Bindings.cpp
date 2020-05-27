// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace TestHelpers::Poisson::dg {

namespace py_bindings {
void bind_first_order(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyPoissonDgTestHelpers, m) {  // NOLINT
  py_bindings::bind_first_order(m);
}

}  // namespace TestHelpers::Poisson::dg
