// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <memory>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <utility>

#include "DataStructures/DenseVector.hpp"
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_dense_vector(py::module& m) {  // NOLINT
  using VectorType = DenseVector<double>;
  py::class_<VectorType>(m, "DenseVector", py::buffer_protocol())
      .def(py::init<size_t>(), py::arg("size"))
      .def(py::init<size_t, double>(), py::arg("size"), py::arg("fill"))
      .def(py::init([](const std::vector<double>& values) {
             VectorType result(values.size());
             std::copy(values.begin(), values.end(), result.begin());
             return result;
           }),
           py::arg("values"))
      .def(py::init([](py::buffer buffer) {
             py::buffer_info info = buffer.request();
             // Sanity-check the buffer
             if (info.format != py::format_descriptor<double>::format()) {
               throw std::runtime_error(
                   "Incompatible format: expected a double array.");
             }
             if (info.ndim != 1) {
               throw std::runtime_error("Incompatible dimension.");
             }
             const auto size = static_cast<size_t>(info.shape[0]);
             auto data = static_cast<double*>(info.ptr);
            VectorType result(size);
            std::copy_n(data, result.size(), result.begin());
            return result;
           }),
           py::arg("buffer"))
      // Expose the data as a Python buffer so it can be cast into Numpy arrays
      .def_buffer([](VectorType& data_vector) {
        return py::buffer_info(data_vector.data(),
                               // Size of one scalar
                               sizeof(double),
                               py::format_descriptor<double>::format(),
                               // Number of dimensions
                               1,
                               // Size of the buffer
                               {data_vector.size()},
                               // Stride for each index (in bytes)
                               {sizeof(double)});
      })
      .def(
          "__iter__",
          [](const VectorType& t) {
            return py::make_iterator(t.begin(), t.end());
          },
          // Keep object alive while iterator exists
          py::keep_alive<0, 1>())
      // __len__ is for being able to write len(my_data_vector) in python
      .def("__len__", &VectorType::size)
      // __getitem__ and __setitem__ are the subscript operators (operator[]).
      // To define (and overload) operator() use __call__
      .def(
          "__getitem__",
          +[](const VectorType& t, const size_t i) {
            bounds_check(t, i);
            return t[i];
          })
      .def(
          "__setitem__",
          +[](VectorType& t, const size_t i, const double v) {
            bounds_check(t, i);
            t[i] = v;
          })
      // Need __str__ for converting to string/printing
      .def(
          "__str__", +[](const VectorType& t) { return get_output(t); })
      // repr allows you to output the object in an interactive python terminal
      // using obj to get the "string REPResenting the object".
      .def(
          "__repr__", +[](const VectorType& t) { return get_output(t); });
}
}  // namespace py_bindings
