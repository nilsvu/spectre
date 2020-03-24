# Distributed under the MIT License.
# See LICENSE.txt for details.

# Look for an executable called sphinx-build or sphinx-build2
find_program(
  SPHINX_EXECUTABLE
  NAMES sphinx-build sphinx-build2
  PATHS ${SPHINX_ROOT}
  DOC "Path to sphinx-build or sphinx-build2 executable")

execute_process(COMMAND "${SPHINX_EXECUTABLE}" "--version"
  RESULT_VARIABLE RESULT
  OUTPUT_VARIABLE OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(RESULT MATCHES 0)
  string(REGEX REPLACE "sphinx-build[2]? " "" SPHINX_VERSION ${OUTPUT})
endif(RESULT MATCHES 0)

set(Sphinx_Version ${SPHINX_VERSION})
set(SPHINX_EXECUTABLE ${SPHINX_EXECUTABLE})

include(FindPackageHandleStandardArgs)
# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Sphinx
  REQUIRED_VARS SPHINX_EXECUTABLE SPHINX_VERSION
  VERSION_VAR SPHINX_VERSION
  )
