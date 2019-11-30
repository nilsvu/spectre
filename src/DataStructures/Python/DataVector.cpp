// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
// #include "PythonBindings/BoundChecks.hpp"
// #include "Utilities/GetOutput.hpp"

namespace py = pybind11;

// namespace {
// std::shared_ptr<DataVector> make_datavector_from_list(bp::list l) {
//   DataVector t = DataVector(static_cast<size_t>(bp::len(l)));
//   for (size_t i = 0; i < static_cast<size_t>(bp::len(l)); i++) {
//     t[i] = bp::extract<double>(l[i]);
//   }
//   return std::make_shared<DataVector>(std::move(t));
// }
// }  // namespace
namespace py_bindings {
void bind_datavector(py::module& m) {
  // Wrapper for basic DataVector operations
  py::class_<DataVector>(m, "DataVector", py::buffer_protocol())
      .def(py::init<size_t>())
      .def(py::init<size_t, double>())
      .def_buffer([](DataVector& data_vector) -> py::buffer_info {
        return py::buffer_info(
            data_vector.data(),                      /* Pointer to buffer */
            sizeof(double),                          /* Size of one scalar */
            py::format_descriptor<double>::format(), /* Python struct-style
                                                       format descriptor */
            1,                                       /* Number of dimensions */
            {data_vector.size()},                    /* Buffer dimensions */
            {sizeof(double)}); /* Strides (in bytes) for each index */
      })
      //  .def("__init__", [](DataVector& data_vector, py::buffer buffer) {
      //    /* Request a buffer descriptor from Python */
      //    py::buffer_info info = buffer.request();

      //    /* Some sanity checks ... */
      //    if (info.format != py::format_descriptor<double>::format())
      //      throw std::runtime_error(
      //          "Incompatible format: expected a double array!");

      //    if (info.ndim != 1)
      //      throw std::runtime_error("Incompatible buffer dimension!");

      //    auto arr = static_cast<double*>(info.ptr);
      //    std::vector<double> vec(static_cast<size_t>(info.shape[0]));
      //    std::copy_n(arr, vec.size(), vec.begin());

      //    new (&data_vector) DataVector(vec);
      //  });
      //  .def("__iter__", py::iterator());
      //  // __len__ is for being able to write len(my_data_vector) in python
      .def("__len__", &DataVector::size);
  //    // __getitem__ and __setitem__ are the subscript operators (operator[]).
  //    // To define (and overload) operator() use __call__
  //    .def("__getitem__",
  //         +[](const DataVector& t, const size_t i) {
  //           bounds_check(t, i);
  //           return t[i];
  //         })
  //    .def("__setitem__",
  //         +[](DataVector& t, const size_t i, const double v) {
  //           bounds_check(t, i);
  //           t[i] = v;
  //         })
  //  // Need __str__ for converting to string/printing
  //  .def("__str__", +[](const DataVector& t) { return get_output(t); })
  //  // repr allows you to output the object in an interactive python terminal
  //  // using obj to get the "string REPResenting the object".
  //  .def("__repr__", +[](const DataVector& t) { return get_output(t); })
  //  .def(bp::self += bp::self)
  //  .def(bp::self += bp::other<double>{})
  //  // Need to do math explicitly converting to DataVector because we don't
  //  // want to represent all the possible expression template types
  //  .def("abs", +[](const DataVector& t) { return DataVector{abs(t)}; })
  //  .def("acos", +[](const DataVector& t) { return DataVector{acos(t)}; })
  //  .def("acosh", +[](const DataVector& t) { return DataVector{acosh(t)}; })
  //  .def("asin", +[](const DataVector& t) { return DataVector{asin(t)}; })
  //  .def("asinh", +[](const DataVector& t) { return DataVector{asinh(t)}; })
  //  .def("atan", +[](const DataVector& t) { return DataVector{atan(t)}; })
  //  .def("atan2",
  //       +[](const DataVector& y, const DataVector& x) {
  //         return DataVector{atan2(y, x)};
  //       })
  //  .def("atanh", +[](const DataVector& t) { return DataVector{atanh(t)}; })
  //  .def("cbrt", +[](const DataVector& t) { return DataVector{cbrt(t)}; })
  //  .def("cos", +[](const DataVector& t) { return DataVector{cos(t)}; })
  //  .def("cosh", +[](const DataVector& t) { return DataVector{cosh(t)}; })
  //  .def("erf", +[](const DataVector& t) { return DataVector{erf(t)}; })
  //  .def("erfc", +[](const DataVector& t) { return DataVector{erfc(t)}; })
  //  .def("exp", +[](const DataVector& t) { return DataVector{exp(t)}; })
  //  .def("exp2", +[](const DataVector& t) { return DataVector{exp2(t)}; })
  //  .def("exp10", +[](const DataVector& t) { return DataVector{exp10(t)}; })
  //  .def("fabs", +[](const DataVector& t) { return DataVector{fabs(t)}; })
  //  .def("hypot",
  //       +[](const DataVector& x, const DataVector& y) {
  //         return DataVector{hypot(x, y)};
  //       })
  //  .def("invcbrt",
  //       +[](const DataVector& t) { return DataVector{invcbrt(t)}; })
  //  .def("invsqrt",
  //       +[](const DataVector& t) { return DataVector{invsqrt(t)}; })
  //  .def("log", +[](const DataVector& t) { return DataVector{log(t)}; })
  //  .def("log2", +[](const DataVector& t) { return DataVector{log2(t)}; })
  //  .def("log10", +[](const DataVector& t) { return DataVector{log10(t)}; })
  //  .def("max", +[](const DataVector& t) { return double{max(t)}; })
  //  .def("min", +[](const DataVector& t) { return double{min(t)}; })
  //  .def("pow", +[](const DataVector& base,
  //                  double exp) { return DataVector{pow(base, exp)}; })
  //  .def("sin", +[](const DataVector& t) { return DataVector{sin(t)}; })
  //  .def("sinh", +[](const DataVector& t) { return DataVector{sinh(t)}; })
  //  .def("sqrt", +[](const DataVector& t) { return DataVector{sqrt(t)}; })
  //  .def("step_function",
  //       +[](const DataVector& t) { return DataVector{step_function(t)}; })
  //  .def("tan", +[](const DataVector& t) { return DataVector{tan(t)}; })
  //  .def("tanh", +[](const DataVector& t) { return DataVector{tanh(t)}; })
  //  .def("__pow__",
  //       +[](const DataVector& base, const double exp) {
  //         return DataVector{pow(base, exp)};
  //       })
  //  .def("__add__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{self + other};
  //       })
  //  .def("__radd__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{other + self};
  //       })
  //  .def("__sub__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{self - other};
  //       })
  //  .def("__rsub__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{other - self};
  //       })
  //  .def("__mul__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{self * other};
  //       })
  //  .def("__rmul__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{other * self};
  //       })
  //  // Need __div__ for python 2 and __truediv__ for python 3.
  //  .def("__div__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{self / other};
  //       })
  //  .def("__truediv__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{self / other};
  //       })
  //  .def("__rdiv__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{other / self};
  //       })
  //  .def("__rtruediv__",
  //       +[](const DataVector& self, const double other) {
  //         return DataVector{other / self};
  //       })

  //  // DataVector-DataVector math
  //  .def("__add__",
  //       +[](const DataVector& self, const DataVector& other) {
  //         return DataVector{self + other};
  //       })
  //  .def("__sub__",
  //       +[](const DataVector& self, const DataVector& other) {
  //         return DataVector{self - other};
  //       })
  //  .def("__mul__",
  //       +[](const DataVector& self, const DataVector& other) {
  //         return DataVector{self * other};
  //       })
  //  .def("__div__",
  //       +[](const DataVector& self, const DataVector& other) {
  //         return DataVector{self / other};
  //       })
  //  .def("__truediv__",
  //       +[](const DataVector& self, const DataVector& other) {
  //         return DataVector{self / other};
  //       })
  //  // NOLINTNEXTLINE(misc-redundant-expression)
  //  .def(bp::self == bp::self)
  //  // NOLINTNEXTLINE(misc-redundant-expression)
  //  .def(bp::self != bp::self)
  //  .def("__neg__", +[](const DataVector& t) { return DataVector{-t}; });
}
}  // namespace py_bindings
