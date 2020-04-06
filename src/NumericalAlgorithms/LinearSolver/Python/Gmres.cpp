// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"

namespace py = pybind11;

namespace LinearSolver {
namespace Serial {
namespace py_bindings {
void bind_gmres(py::module& m) {  // NOLINT
  using OperandType = DenseVector<double>;
  using OperatorType = DenseMatrix<double>;
  py::class_<Gmres<OperandType>>(m, "Gmres")
      .def(py::init([](const size_t max_iterations,
                       const double absolute_residual,
                       const double relative_residual, const size_t restart) {
             return Gmres<OperandType>{
                 Convergence::Criteria{max_iterations, absolute_residual,
                                       relative_residual},
                 ::Verbosity::Quiet, restart};
           }),
           py::arg("max_iterations"), py::arg("absolute_residual"),
           py::arg("relative_residual"), py::arg("restart") = 0)
      .def("__call__",
           [](const Gmres<OperandType>& gmres,
              const OperatorType& linear_operator, const OperandType& source,
              const OperandType& initial_guess) {
             return gmres(
                 [&linear_operator](const OperandType& arg) {
                   return OperandType{linear_operator * arg};
                 },
                 source, initial_guess);
           });
}
}  // namespace py_bindings
}  // namespace Serial
}  // namespace LinearSolver
