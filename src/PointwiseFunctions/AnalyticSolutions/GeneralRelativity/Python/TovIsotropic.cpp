// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <sstream>
#include <string>
#include <utility>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TovIsotropic.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"

namespace bp = boost::python;

namespace gr {
namespace Solutions {
namespace py_bindings {

void bind_tov_isotropic() {
  bp::class_<TovIsotropic, boost::noncopyable>(
      "TovIsotropic",
      bp::init<EquationsOfState::PolytropicFluid<true>, double>())
      .def("outer_radius", &TovIsotropic::outer_radius)
      .def("mass", &TovIsotropic::mass)
      .def("conformal_factor", &TovIsotropic::conformal_factor)
      .def("log_specific_enthalpy", &TovIsotropic::log_specific_enthalpy);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace gr
