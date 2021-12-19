// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>

#include "Elliptic/DiscontinuousGalerkin/Python/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/BuildOperatorMatrix.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace elliptic::dg::py_bindings {

namespace detail {
template <size_t Dim>
void bind_elasticity_operator_impl(py::module& m) {  // NOLINT
  using system = Elasticity::FirstOrderSystem<Dim>;
  m.def(("build_elasticity_operator_matrix_" + get_output(Dim) + "d").c_str(),
        [](const DomainCreator<Dim>& domain_creator,
           const double penalty_parameter, const bool massive) {
          return build_operator_matrix<system, true>(
              domain_creator, penalty_parameter, massive,
              tuples::TaggedTuple<
                  Elasticity::Tags::ConstitutiveRelation<Dim>>{std::make_unique<
                  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(
                  1., 2.)});
        },
        py::arg("domain_creator"), py::arg("penalty_parameter"),
        py::arg("massive"));
}
}  // namespace detail

void bind_elasticity_operator(py::module& m) {  // NOLINT
  detail::bind_elasticity_operator_impl<2>(m);
  detail::bind_elasticity_operator_impl<3>(m);
}

}  // namespace elliptic::dg::py_bindings
