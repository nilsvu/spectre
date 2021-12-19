// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <string>

#include "Elliptic/DiscontinuousGalerkin/Python/BuildOperatorMatrix.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace elliptic::dg::py_bindings {

namespace detail {
template <size_t Dim>
void bind_poisson_operator_impl(py::module& m) {  // NOLINT
  using system =
      Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;
  m.def(("build_poisson_operator_matrix_" + get_output(Dim) + "d").c_str(),
        [](const DomainCreator<Dim>& domain_creator,
           const double penalty_parameter, const bool massive) {
          return build_operator_matrix<system, true>(
              domain_creator, penalty_parameter, massive);
        },
        py::arg("domain_creator"), py::arg("penalty_parameter"),
        py::arg("massive"));
}
}  // namespace detail

void bind_poisson_operator(py::module& m) {  // NOLINT
  detail::bind_poisson_operator_impl<1>(m);
  detail::bind_poisson_operator_impl<2>(m);
  detail::bind_poisson_operator_impl<3>(m);
}

}  // namespace elliptic::dg::py_bindings
