// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"

namespace py = pybind11;

namespace Xcts {
namespace Solutions {
namespace py_bindings {

void bind_constant_density_star(py::module& m) {  // NOLINT
  py::class_<ConstantDensityStar>(m, "ConstantDensityStar")
      .def(py::init<double, double>(), py::arg("density"), py::arg("radius"))
      .def("density", &ConstantDensityStar::density)
      .def("radius", &ConstantDensityStar::radius);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Xcts
