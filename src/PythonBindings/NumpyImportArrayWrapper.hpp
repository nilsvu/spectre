// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// The Numpy API changed between Python2 and Python3: The `import_array` macro
// returns NULL in Python3, but does not return in Python2. Therefore, we wrap
// the macro in this function and include it in each module's `Numpy.hpp`.
// See `DataStructures/Python/Numpy.hpp` for an example.
#if PY_MAJOR_VERSION >= 3
std::nullptr_t
#else
void
#endif
spectre_numpy_import_array() {
  import_array();
#if PY_MAJOR_VERSION >= 3
  return nullptr;
#endif
}
