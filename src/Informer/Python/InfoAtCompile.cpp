// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>
#include <string>
#include <utility>

#include "Informer/InfoFromBuild.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_info_at_compile(py::module& m) {
  // Wrapper to make Build Info available from python
  m.def("spectre_version", &spectre_version);
  m.def("spectre_major_version", &spectre_major_version);
  m.def("spectre_minor_version", &spectre_minor_version);
  m.def("spectre_patch_version", &spectre_patch_version);
  m.def("unit_test_path", &unit_test_path);
}
}  // namespace py_bindings
