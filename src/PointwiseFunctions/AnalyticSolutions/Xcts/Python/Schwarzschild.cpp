// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace Xcts {
namespace Solutions {
namespace py_bindings {

template <SchwarzschildCoordinates Coords>
void bind_schwarzschild_impl(py::module& m) {  // NOLINT
  py::class_<Schwarzschild<Coords>>(
      m, ("Schwarzschild" + get_output(Coords)).c_str())
      .def(py::init<>())
      .def("radius_at_horizon", &Schwarzschild<Coords>::radius_at_horizon);
}

void bind_schwarzschild(py::module& m) {  // NOLINT
  bind_schwarzschild_impl<SchwarzschildCoordinates::Isotropic>(m);
  bind_schwarzschild_impl<SchwarzschildCoordinates::KerrSchildIsotropic>(m);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Xcts
