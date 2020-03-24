# Distributed under the MIT License.
# See LICENSE.txt for details.

# Look for an executable called breathe-apidoc
find_program(
  BREATHE_APIDOC_EXECUTABLE
  NAMES breathe-apidoc
  PATHS ${BREATHE_APIDOC_ROOT}
  DOC "Path to breathe-apidoc executable")

execute_process(COMMAND "${BREATHE_APIDOC_EXECUTABLE}" "--version"
  RESULT_VARIABLE RESULT
  OUTPUT_VARIABLE OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

include(FindPackageHandleStandardArgs)

if(RESULT MATCHES 0)
  string(REGEX MATCH "[0123456789]+\.[0123456789]+[\.]?[0123456789]*"
    BREATHE_APIDOC_VERSION ${OUTPUT})
  set(BREATHE_VERSION ${BREATHE_APIDOC_VERSION})
endif(RESULT MATCHES 0)

# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Breathe
  REQUIRED_VARS BREATHE_APIDOC_EXECUTABLE BREATHE_VERSION
  BREATHE_APIDOC_VERSION
  VERSION_VAR BREATHE_VERSION)
