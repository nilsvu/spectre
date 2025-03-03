// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/CircularOrbit.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace ScalarSelfForce::AnalyticData::py_bindings {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");

  py::class_<CircularOrbit>(m, "CircularOrbit")
      .def(py::init<double, double, double, int>(), py::arg("black_hole_mass"),
           py::arg("black_hole_spin"), py::arg("orbital_radius"),
           py::arg("m_mode_number"))
      .def_property_readonly("black_hole_mass", &CircularOrbit::black_hole_mass)
      .def_property_readonly("black_hole_spin", &CircularOrbit::black_hole_spin)
      .def_property_readonly("orbital_radius", &CircularOrbit::orbital_radius)
      .def_property_readonly("m_mode_number", &CircularOrbit::m_mode_number)
      .def("puncture_position", &CircularOrbit::puncture_position);
}

}  // namespace ScalarSelfForce::AnalyticData::py_bindings
