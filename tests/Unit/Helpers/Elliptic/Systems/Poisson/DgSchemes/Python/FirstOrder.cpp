// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Matrix.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Helpers/Elliptic/Systems/Poisson/DgSchemes/FirstOrder.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace TestHelpers::Poisson::dg::py_bindings {

namespace detail {
template <size_t Dim>
void bind_first_order(py::module& m) {  // NOLINT
  m.def(("first_order_operator_matrix_" + get_output(Dim) + "d").c_str(),
        &first_order_operator_matrix<Dim>);
}
}  // namespace detail

void bind_first_order(py::module& m) {  // NOLINT
  detail::bind_first_order<1>(m);
  detail::bind_first_order<2>(m);
  detail::bind_first_order<3>(m);
}

}  // namespace TestHelpers::Poisson::dg::py_bindings
