// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// The Numpy API changed between Python2 and Python3: The `import_array` macro
// returns NULL in Python3, but does not return in Python2. Therefore, we wrap
// the macro in this function and include it in each module's `Numpy.hpp`.
// See `DataStructures/Python/Numpy.hpp` for an example.
std::nullptr_t
spectre_numpy_import_array() {
  import_array();
  return nullptr;
}
