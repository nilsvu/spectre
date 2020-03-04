// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace Poisson {
namespace Solutions {
namespace py_bindings {
void bind_product_of_sinusoids(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyPoissonSolutions, m) {  // NOLINT
  py_bindings::bind_product_of_sinusoids(m);
}

}  // namespace Solutions
}  // namespace Poisson
