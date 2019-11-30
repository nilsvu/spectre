# Distributed under the MIT License.
# See LICENSE.txt for details.

find_path(
    PYBIND11_INCLUDE_DIR
    PATH_SUFFIXES include
    NAMES pybind11/pybind11.h
    HINTS ${PYBIND11_ROOT}
    DOC "Pybind11 include directory. Used PYBIND11_ROOT to set a search dir."
)

set(PYBIND11_INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Pybind11
    DEFAULT_MSG PYBIND11_INCLUDE_DIR PYBIND11_INCLUDE_DIRS
)
mark_as_advanced(PYBIND11_INCLUDE_DIR PYBIND11_INCLUDE_DIRS)
