// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace TestHelpers {
namespace Poisson {
namespace Solutions {

namespace py_bindings {
void bind_verify_solution(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyPoissonSolutionsTestHelpers, m) {  // NOLINT
  py_bindings::bind_verify_solution(m);
}

}  // namespace Solutions
}  // namespace Poisson
}  // namespace TestHelpers
