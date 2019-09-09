// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <sstream>
#include <string>
#include <utility>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace bp = boost::python;

namespace gr {
namespace Solutions {
namespace py_bindings {
void bind_tov() {
  bp::class_<TovSolution, boost::noncopyable>(
      "Tov", bp::init<EquationsOfState::EquationOfState<true, 1>, double>())
      .def("outer_radius", &TovSolution::outer_radius)
      .def("mass", &TovSolution::mass)
      .def("log_specific_enthalpy", &TovSolution::log_specific_enthalpy);
}
}  // namespace py_bindings
}  // namespace Solutions
}  // namespace gr
