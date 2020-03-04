// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace Poisson {
namespace Solutions {
namespace py_bindings {

namespace {
template <size_t Dim>
void bind_product_of_sinusoids_impl(py::module& m) {  // NOLINT
  py::class_<ProductOfSinusoids<Dim>>(
      m, ("ProductOfSinusoids" + get_output(Dim) + "D").c_str())
      .def(py::init<std::array<double, Dim>>(), py::arg("wave_numbers"));
}
}  // namespace

void bind_product_of_sinusoids(py::module& m) {  // NOLINT
  bind_product_of_sinusoids_impl<1>(m);
  bind_product_of_sinusoids_impl<2>(m);
  bind_product_of_sinusoids_impl<3>(m);
}
}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Poisson
