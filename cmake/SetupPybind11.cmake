# Distributed under the MIT License.
# See LICENSE.txt for details.

if(BUILD_PYTHON_BINDINGS)
  # Make sure to find the Python interpreter first, so it is consistent with
  # the one that pybind11 uses
  find_package(PythonInterp)
  find_package(PythonLibs)
  find_package(Pybind11 REQUIRED)

  spectre_include_directories("${PYBIND11_INCLUDE_DIR}")
  message(STATUS "Pybind11 include: ${PYBIND11_INCLUDE_DIR}")
endif()
