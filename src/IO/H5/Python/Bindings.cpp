// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>

#define SPECTRE_NUMPY_IMPORT_ARRAY
#include "IO/H5/Python/Numpy.hpp"

namespace py_bindings {
void bind_h5file();
void bind_h5dat();
void bind_h5vol();
}  // namespace py_bindings

BOOST_PYTHON_MODULE(_PyH5) {
  Py_Initialize();
  spectre_numpy_import_array();
  py_bindings::bind_h5file();
  py_bindings::bind_h5dat();
  py_bindings::bind_h5vol();
}
