// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <sstream>
#include <string>
#include <utility>

#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"

namespace bp = boost::python;

namespace EquationsOfState {
namespace py_bindings {

// class SomeClass : public PUP::able {
//  public:
//   SomeClass() {}
//   // PUP::able support: decl, migration constructor, and pup
//   PUPable_decl(SomeClass);
//   SomeClass(CkMigrateMessage* m) : PUP::able(m) {}
//   virtual void pup(PUP::er& p) {
//     PUP::able::pup(p);  // Call base class
//   }
// };

// class DerivedClass : public SomeClass {
//  public:
//   DerivedClass() {}
//   DerivedClass(int x) : x_(x) {}

//  private:
//   int x_ = 1;
// };

void bind_polytropic_fluid() {
  bp::class_<PUP::able, boost::noncopyable>("PUPable", bp::no_init);
  // bp::class_<DerivedClass, boost::noncopyable>("DerivedClass",
  // bp::init<int>()); bp::class_<PolytropicFluid<true>, boost::noncopyable>(
  //     "PolytropicFluid", bp::init<double, double>());
}
}  // namespace py_bindings
}  // namespace EquationsOfState
