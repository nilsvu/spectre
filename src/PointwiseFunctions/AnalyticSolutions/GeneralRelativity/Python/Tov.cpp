// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"

namespace bp = boost::python;

namespace gr {
namespace Solutions {
namespace py_bindings {

void bind_tov() {
  bp::class_<TovSolution, boost::noncopyable>(
      "Tov",
      // Only allowing a polytropic EOS here for now since boost::python has
      // trouble with virtual base classes.
      bp::init<EquationsOfState::PolytropicFluid<true>, double>())
      .def("outer_radius", &TovSolution::outer_radius)
      .def("mass", &TovSolution::mass)
      .def("log_specific_enthalpy", &TovSolution::log_specific_enthalpy);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace gr
