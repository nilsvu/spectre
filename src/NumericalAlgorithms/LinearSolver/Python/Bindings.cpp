// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace LinearSolver {
namespace Serial {
namespace py_bindings {
void bind_gmres(py::module& m);  // NOLINT
}  // namespace py_bindings
}  // namespace Serial
}  // namespace LinearSolver

PYBIND11_MODULE(_PyDomainCreators, m) {  // NOLINT
  LinearSolver::Serial::py_bindings::bind_gmres(m);
}
