// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// These macros are required so that the NumPy API will work when used in
// multiple cpp files.  See
// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#ifdef PY_ARRAY_UNIQUE_SYMBOL
static_assert(false, "Already have a PY_ARRAY_UNIQUE_SYMBOL defined.");
#endif
#define PY_ARRAY_UNIQUE_SYMBOL SPECTRE_IO_H5_PYTHON_BINDINGS
// Code is clean against Numpy 1.7.  See
// https://docs.scipy.org/doc/numpy-1.15.1/reference/c-api.deprecations.html
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Define this macro before including this file in the `Bindings.cpp`
// so you can call `spectre_numpy_import_array()` when setting up the module.
#ifdef SPECTRE_NUMPY_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "PythonBindings/NumpyImportArrayWrapper.hpp"
#else
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#endif
