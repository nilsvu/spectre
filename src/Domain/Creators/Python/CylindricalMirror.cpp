// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/CylindricalMirror.hpp"
#include "Domain/Creators/DomainCreator.hpp"

namespace py = pybind11;

namespace domain {
namespace creators {
namespace py_bindings {
void bind_cylindrical_mirror(py::module& m) {  // NOLINT
  py::class_<CylindricalMirror, DomainCreator<3>>(m, "CylindricalMirror")
      .def(py::init<std::vector<double>, std::vector<size_t>,
                    std::vector<std::array<size_t, 2>>,
                    std::vector<std::vector<size_t>>,
                    std::vector<std::vector<size_t>>,
                    std::vector<std::vector<size_t>>,
                    std::vector<std::vector<size_t>>, std::vector<double>,
                    std::vector<std::vector<size_t>>,
                    std::vector<std::vector<size_t>>, bool>(),
           py::arg("PartitioningInR"), py::arg("Square_refinement"),
           py::arg("Square_gridpoints"), py::arg("R_refinement"),
           py::arg("R_gridpoints"), py::arg("Theta_refinement"),
           py::arg("Theta_gridpoints"), py::arg("PartitioningInZ"),
           py::arg("Z_refinement"), py::arg("Z_gridpoints"),
           py::arg("use_equiangular_map"));
}
}  // namespace py_bindings
}  // namespace creators
}  // namespace domain
