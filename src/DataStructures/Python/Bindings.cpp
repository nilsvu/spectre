// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>

#define SPECTRE_NUMPY_IMPORT_ARRAY
#include "DataStructures/Python/Numpy.hpp"

namespace py_bindings {
void bind_datavector();
void bind_matrix();
}  // namespace py_bindings

BOOST_PYTHON_MODULE(_DataStructures) {
  Py_Initialize();
  spectre_numpy_import_array();
  py_bindings::bind_datavector();
  py_bindings::bind_matrix();
}
