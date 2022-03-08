# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_define_test_timeout_factor_option(PYTHON "Python")

option(PY_DEV_MODE "Enable development mode for the Python package, meaning \
that Python files are symlinked rather than copied to the build directory" OFF)
function(configure_or_symlink_py_file SOURCE_FILE TARGET_FILE)
  if(PY_DEV_MODE)
    get_filename_component(_TARGET_FILE_DIR ${TARGET_FILE} DIRECTORY)
    execute_process(COMMAND
      ${CMAKE_COMMAND} -E make_directory ${_TARGET_FILE_DIR})
    execute_process(COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${SOURCE_FILE} ${TARGET_FILE})
  else()
    configure_file(${SOURCE_FILE} ${TARGET_FILE})
  endif()
endfunction()

set(SPECTRE_PYTHON_PREFIX "${CMAKE_BINARY_DIR}/bin/python/spectre/")
get_filename_component(
  SPECTRE_PYTHON_PREFIX
  "${SPECTRE_PYTHON_PREFIX}" ABSOLUTE)
get_filename_component(
  SPECTRE_PYTHON_PREFIX_PARENT
  "${SPECTRE_PYTHON_PREFIX}/.." ABSOLUTE)

message(STATUS "Configure Python package in: ${SPECTRE_PYTHON_PREFIX_PARENT}")
message(STATUS "Install it with: pip install -e ${SPECTRE_PYTHON_PREFIX_PARENT}")

# Create the root __init__.py file
if(NOT EXISTS "${SPECTRE_PYTHON_PREFIX}/__init__.py")
  file(WRITE
    "${SPECTRE_PYTHON_PREFIX}/__init__.py"
    "__all__ = []")
endif()

# Create the root __main__.py entry point
configure_or_symlink_py_file(
  "${CMAKE_SOURCE_DIR}/support/Python/__main__.py"
  "${SPECTRE_PYTHON_PREFIX}/__main__.py"
)
# Also link the main entry point to bin/
set(PYTHON_SCRIPT_LOCATION "spectre")
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/SpectrePythonExecutable.sh"
  "${CMAKE_BINARY_DIR}/tmp/spectre")
file(COPY "${CMAKE_BINARY_DIR}/tmp/spectre"
  DESTINATION "${CMAKE_BINARY_DIR}/bin"
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
    GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Write configuration files for installing the Python modules
configure_or_symlink_py_file(
  "${CMAKE_SOURCE_DIR}/src/PythonBindings/pyproject.toml"
  "${SPECTRE_PYTHON_PREFIX_PARENT}/pyproject.toml")
configure_or_symlink_py_file(
  "${CMAKE_SOURCE_DIR}/src/PythonBindings/setup.cfg"
  "${SPECTRE_PYTHON_PREFIX_PARENT}/setup.cfg")

set(_JEMALLOC_MESSAGE "")
if(BUILD_PYTHON_BINDINGS AND "${JEMALLOC_LIB_TYPE}" STREQUAL SHARED)
  set(_JEMALLOC_MESSAGE
    "echo 'You must run python as:'\n"
    "echo 'LD_PRELOAD=${JEMALLOC_LIBRARIES} python ...'\n")
  string(REPLACE ";" "" _JEMALLOC_MESSAGE "${_JEMALLOC_MESSAGE}")
endif()

# Write a file to be able to set up the new python path.
file(WRITE
  "${CMAKE_BINARY_DIR}/tmp/LoadPython.sh"
  "#!/bin/sh\n"
  "export PYTHONPATH=$PYTHONPATH:${SPECTRE_PYTHON_PREFIX_PARENT}\n"
  ${_JEMALLOC_MESSAGE}
  )
configure_file(
  "${CMAKE_BINARY_DIR}/tmp/LoadPython.sh"
  "${CMAKE_BINARY_DIR}/bin/LoadPython.sh")

# Install the SpECTRE Python package to the CMAKE_INSTALL_PREFIX, using pip.
# This will install the package into the expected subdirectory, typically
# `lib/pythonX.Y/site-packages/`. It also creates symlinks to entry points
# specified in `setup.py`.
install(
  CODE "execute_process(\
    COMMAND ${Python_EXECUTABLE} -m pip install \
      --no-deps --no-input --no-cache-dir --no-index --ignore-installed \
      --disable-pip-version-check --no-build-isolation \
      --prefix ${CMAKE_INSTALL_PREFIX} ${SPECTRE_PYTHON_PREFIX_PARENT} \
    )"
  )

add_custom_target(all-pybindings)

# Add a python module, either with or without python bindings and with
# or without additional python files. If bindings are being provided then
# the library will be named Py${MODULE_NAME}, e.g. if MODULE_NAME is
# DataStructures then the library name is PyDataStructures.
#
# - MODULE_NAME   The name of the module, e.g. DataStructures.
#
# - MODULE_PATH   Path inside the module, e.g. submodule0/submodule1 would
#                 result in loading spectre.submodule0.submodule1
#
# - SOURCES       The C++ source files for bindings. Omit if no bindings
#                 are being generated.
#
# - LIBRARY_NAME  The name of the C++ libray, e.g. PyDataStructures.
#                 Required if SOURCES are specified. Must begin with "Py".
#
# - PYTHON_FILES  List of the python files (relative to
#                 ${CMAKE_SOURCE_DIR}/src) to add to the module. Omit if
#                 no python files are to be provided.
#
# - PYTHON_EXECUTABLES  List of additional python files that should be exposed
#                 as executables. Same format as PYTHON_FILES. The only
#                 difference is that these scripts can be run directly from
#                 `${CMAKE_BINARY_DIR}/bin`. Note that any Python script can be
#                 executed from its location in the Python package, even if it's
#                 not listed in PYTHON_EXECUTABLES.
function(SPECTRE_PYTHON_ADD_MODULE MODULE_NAME)
  if(BUILD_PYTHON_BINDINGS AND
      "${JEMALLOC_LIB_TYPE}" STREQUAL STATIC
      AND BUILD_SHARED_LIBS)
    message(FATAL_ERROR
      "Cannot build python bindings when using a static library JEMALLOC and "
      "building SpECTRE with shared libraries. Either disable the python "
      "bindings using -D BUILD_PYTHON_BINDINGS=OFF, switch to a shared/dynamic "
      "JEMALLOC library, use the system allocator by passing "
      "-D MEMORY_ALLOCATOR=SYSTEM to CMake, or build SpECTRE using static "
      "libraries by passing -D BUILD_SHARED_LIBS=OFF to CMake.")
  endif()

  set(SINGLE_VALUE_ARGS MODULE_PATH LIBRARY_NAME)
  set(MULTI_VALUE_ARGS SOURCES PYTHON_FILES PYTHON_EXECUTABLES)
  cmake_parse_arguments(
    ARG ""
    "${SINGLE_VALUE_ARGS}"
    "${MULTI_VALUE_ARGS}"
    ${ARGN})
  list(APPEND ARG_PYTHON_FILES ${ARG_PYTHON_EXECUTABLES})

  set(MODULE_LOCATION
    "${SPECTRE_PYTHON_PREFIX}/${ARG_MODULE_PATH}/${MODULE_NAME}")
  get_filename_component(MODULE_LOCATION ${MODULE_LOCATION} ABSOLUTE)

  # Module location in Python syntax, e.g. "spectre.Visualization"
  string(REPLACE "/" ";" PYTHON_MODULE_LOCATION_COMPONENTS "${ARG_MODULE_PATH}")
  list(INSERT PYTHON_MODULE_LOCATION_COMPONENTS 0 "spectre")
  list(APPEND PYTHON_MODULE_LOCATION_COMPONENTS "${MODULE_NAME}")
  list(JOIN PYTHON_MODULE_LOCATION_COMPONENTS "." PYTHON_MODULE_LOCATION)

  # Create list of all the python submodule names
  set(PYTHON_SUBMODULES "")
  foreach(PYTHON_FILE ${ARG_PYTHON_FILES})
    # Get file name Without Extension (NAME_WE)
    get_filename_component(PYTHON_FILE "${PYTHON_FILE}" NAME_WE)
    list(APPEND PYTHON_SUBMODULES ${PYTHON_FILE})
  endforeach(PYTHON_FILE ${ARG_PYTHON_FILES})

  # Add our python library, if it has sources
  set(SPECTRE_PYTHON_MODULE_IMPORT "")
  if(BUILD_PYTHON_BINDINGS AND NOT "${ARG_SOURCES}" STREQUAL "")
    if("${ARG_LIBRARY_NAME}" STREQUAL "")
      message(FATAL_ERROR "Set a LIBRARY_NAME for Python module "
          "'${MODULE_NAME}' that has sources.")
    endif()
    if(NOT "${ARG_LIBRARY_NAME}" MATCHES "^Py")
      message(FATAL_ERROR "The LIBRARY_NAME for Python module "
          "'${MODULE_NAME}' must begin with 'Py' but is '${ARG_LIBRARY_NAME}'.")
    endif()

    Python_add_library(${ARG_LIBRARY_NAME} MODULE ${ARG_SOURCES})
    set_target_properties(
      ${ARG_LIBRARY_NAME}
      PROPERTIES
      # These can be turned on once we support them
      INTERPROCEDURAL_OPTIMIZATION OFF
      CXX__VISIBILITY_PRESET OFF
      VISIBLITY_INLINES_HIDDEN OFF
      )
    # In order to avoid runtime errors about missing compatibility functions
    # defined in the PythonBindings library, we need to link in the whole
    # archive. This is not needed on macOS.
    if (APPLE)
      target_link_libraries(
        ${ARG_LIBRARY_NAME}
        PUBLIC PythonBindings
        )
    else()
      target_link_libraries(
        ${ARG_LIBRARY_NAME}
        PUBLIC
        -Wl,--whole-archive
        PythonBindings
        -Wl,--no-whole-archive
        )
    endif()
    target_link_libraries(
      ${ARG_LIBRARY_NAME}
      PRIVATE
      CharmModuleInit
      SpectreFlags
      )
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      # Clang doesn't by default enable sized deallocation so we need to
      # enable it explicitly. This can potentially cause problems if the
      # standard library being used is too old, but GCC doesn't have any
      # safeguards against that either.
      #
      # See https://github.com/pybind/pybind11/issues/1604
      target_compile_options(${ARG_LIBRARY_NAME}
        PRIVATE -fsized-deallocation)
    endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # We don't want the 'lib' prefix for python modules, so we set the output
    # name
    set_target_properties(
      ${ARG_LIBRARY_NAME}
      PROPERTIES
      PREFIX ""
      LIBRARY_OUTPUT_NAME "_${ARG_LIBRARY_NAME}"
      LIBRARY_OUTPUT_DIRECTORY ${MODULE_LOCATION}
      )
    # We need --no-as-needed since each python module needs to depend on all the
    # shared libraries in order to run successfully.
    set(PY_LIB_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS}")
    if(NOT APPLE)
      set(PY_LIB_LINK_FLAGS
        "${CMAKE_CXX_LINK_FLAGS} -Wl,--no-as-needed")
    endif()
    set_target_properties(
      ${ARG_LIBRARY_NAME}
      PROPERTIES
      LINK_FLAGS "${PY_LIB_LINK_FLAGS}"
      )
    set(SPECTRE_PYTHON_MODULE_IMPORT "from ._${ARG_LIBRARY_NAME} import *")
    if(BUILD_TESTING)
      add_dependencies(test-executables ${ARG_LIBRARY_NAME})
    endif()
    add_dependencies(all-pybindings ${ARG_LIBRARY_NAME})
  endif(BUILD_PYTHON_BINDINGS AND NOT "${ARG_SOURCES}" STREQUAL "")

  # Read the __init__.py file if it exists
  set(INIT_FILE_LOCATION "${MODULE_LOCATION}/__init__.py")
  set(INIT_FILE_CONTENTS "")
  if(EXISTS ${INIT_FILE_LOCATION})
    file(READ
      ${INIT_FILE_LOCATION}
      INIT_FILE_CONTENTS)
  endif(EXISTS ${INIT_FILE_LOCATION})

  # Update the "from ._LIB import *" in the __init__.py
  if("${INIT_FILE_CONTENTS}" STREQUAL "")
    set(INIT_FILE_OUTPUT "${SPECTRE_PYTHON_MODULE_IMPORT}\n__all__ = []\n")
  else("${INIT_FILE_CONTENTS}" STREQUAL "")
    string(FIND ${INIT_FILE_CONTENTS} "from" FOUND_FROM_STATEMENT)
    if (${FOUND_FROM_STATEMENT} EQUAL -1)
      set(INIT_FILE_OUTPUT
        "${SPECTRE_PYTHON_MODULE_IMPORT}\n${INIT_FILE_CONTENTS}")
    else()
      string(REGEX REPLACE
        "from[^\n]+"
        "${SPECTRE_PYTHON_MODULE_IMPORT}"
        INIT_FILE_OUTPUT
        ${INIT_FILE_CONTENTS})
    endif()
  endif("${INIT_FILE_CONTENTS}" STREQUAL "")

  # configure the source files into the build directory and make sure
  # they are in the init file.
  foreach(PYTHON_FILE ${ARG_PYTHON_FILES})
    # Configure file
    get_filename_component(PYTHON_FILE_JUST_NAME
      "${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_FILE}" NAME)
    configure_or_symlink_py_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_FILE}"
      "${MODULE_LOCATION}/${PYTHON_FILE_JUST_NAME}"
      )

    # Update init file
    get_filename_component(PYTHON_FILE_JUST_NAME_WE
      "${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_FILE}" NAME_WE)
    string(FIND
      ${INIT_FILE_OUTPUT}
      "\"${PYTHON_FILE_JUST_NAME_WE}\""
      INIT_FILE_CONTAINS_ME)
    if(${INIT_FILE_CONTAINS_ME} EQUAL -1)
      string(REPLACE
        "__all__ = ["
        "__all__ = [\"${PYTHON_FILE_JUST_NAME_WE}\", "
        INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})
    endif(${INIT_FILE_CONTAINS_ME} EQUAL -1)

    # Write an executable in `${CMAKE_BINARY_DIR}/bin` that runs the Python
    # script in the correct Python environment
    if(${PYTHON_FILE} IN_LIST ARG_PYTHON_EXECUTABLES)
      set(PYTHON_SCRIPT_LOCATION
        "${PYTHON_MODULE_LOCATION}.${PYTHON_FILE_JUST_NAME_WE}")
      configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/SpectrePythonExecutable.sh"
        "${CMAKE_BINARY_DIR}/tmp/${PYTHON_FILE_JUST_NAME_WE}")
      file(COPY "${CMAKE_BINARY_DIR}/tmp/${PYTHON_FILE_JUST_NAME_WE}"
        DESTINATION "${CMAKE_BINARY_DIR}/bin"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
          GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    endif()
  endforeach(PYTHON_FILE ${ARG_PYTHON_FILES})

  string(REPLACE ", ]" "]" INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})

  # Remove python files that we are no longer using
  string(REGEX MATCH "\\\[.*\\\]"
    WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS ${INIT_FILE_OUTPUT})
  string(REPLACE "[" "" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  string(REPLACE "]" "" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  string(REPLACE "\"" "" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  string(REPLACE ", " ";" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  foreach(CURRENT_PYTHON_MODULE in ${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS})
    string(FIND "${ARG_PYTHON_FILES}" "${CURRENT_PYTHON_MODULE}.py"
      PYTHON_MODULE_EXISTS)
    if(${PYTHON_MODULE_EXISTS} EQUAL -1
         AND NOT EXISTS "${MODULE_LOCATION}/${CURRENT_PYTHON_MODULE}")
      string(REPLACE "\"${CURRENT_PYTHON_MODULE}\""
        ""
        INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})
      string(REPLACE ", ," "," INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})
      file(REMOVE "${MODULE_LOCATION}/${CURRENT_PYTHON_MODULE}.py")
    endif()
  endforeach(CURRENT_PYTHON_MODULE in ${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS})
  # Sometimes we get a "[," in the files, which can give problems.
  string(REPLACE "[," "[" INIT_FILE_OUTPUT "${INIT_FILE_OUTPUT}")

  # Write the __init__.py file for the module
  if(NOT ${INIT_FILE_OUTPUT} STREQUAL "${INIT_FILE_CONTENTS}")
    file(WRITE ${INIT_FILE_LOCATION} ${INIT_FILE_OUTPUT})
  endif(NOT ${INIT_FILE_OUTPUT} STREQUAL "${INIT_FILE_CONTENTS}")

  # Register with parent submodules:
  # We walk up the tree until we get to ${SPECTRE_PYTHON_PREFIX}
  # and make sure we have all the submodules registered.
  set(CURRENT_MODULE ${MODULE_LOCATION})
  while(NOT ${CURRENT_MODULE} STREQUAL ${SPECTRE_PYTHON_PREFIX})
    get_filename_component(PARENT_MODULE "${CURRENT_MODULE}/.." ABSOLUTE)
    string(REPLACE "${PARENT_MODULE}/" ""
      CURRENT_MODULE_NAME ${CURRENT_MODULE})

    set(PARENT_MODULE_CONTENTS "__all__ = []")
    if(EXISTS "${PARENT_MODULE}/__init__.py")
      file(READ
        "${PARENT_MODULE}/__init__.py"
        PARENT_MODULE_CONTENTS)
    endif(EXISTS "${PARENT_MODULE}/__init__.py")

    string(FIND "${PARENT_MODULE_CONTENTS}" "\"${CURRENT_MODULE_NAME}\""
      PARENT_MODULE_CONTAINS_ME)

    if(${PARENT_MODULE_CONTAINS_ME} EQUAL -1)
      string(REPLACE "__all__ = [" "__all__ = [\"${CURRENT_MODULE_NAME}\", "
        PARENT_MODULE_NEW_CONTENTS "${PARENT_MODULE_CONTENTS}")
      string(REPLACE ", ]" "]"
        PARENT_MODULE_NEW_CONTENTS "${PARENT_MODULE_NEW_CONTENTS}")
      file(WRITE
        "${PARENT_MODULE}/__init__.py"
        ${PARENT_MODULE_NEW_CONTENTS})
    endif(${PARENT_MODULE_CONTAINS_ME} EQUAL -1)

    set(CURRENT_MODULE ${PARENT_MODULE})
  endwhile(NOT ${CURRENT_MODULE} STREQUAL ${SPECTRE_PYTHON_PREFIX})
endfunction()

# Link with the LIBRARIES if Python bindings are being built
function (spectre_python_link_libraries LIBRARY_NAME)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
  target_link_libraries(
    ${LIBRARY_NAME}
    # Forward all remaining arguments
    ${ARGN}
    )
endfunction()

# Add the DEPENDENCIES if Python bindings are being built
function (spectre_python_add_dependencies LIBRARY_NAME)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
  add_dependencies(
    ${LIBRARY_NAME}
    # Forward all remaining arguments
    ${ARGN}
    )
endfunction()

# Register a python test file with ctest.
# - TEST_NAME    The name of the test,
#                e.g. "Unit.DataStructures.Python.DataVector"
#
# - FILE         The file to add, e.g. Test_DataVector.py
#
# - TAGS         A semicolon separated list of labels for the test,
#                e.g. "Unit;DataStructures;Python"
# - PY_MODULE_DEPENDENCY
#                The python module that this test depends on
#                Set to None if there is no python module dependency
function(SPECTRE_ADD_PYTHON_TEST TEST_NAME FILE TAGS
    PY_MODULE_DEPENDENCY)
  get_filename_component(FILE "${FILE}" ABSOLUTE)
  string(TOLOWER "${TAGS}" TAGS)

  add_test(
    NAME "\"${TEST_NAME}\""
    COMMAND
    ${Python_EXECUTABLE}
    ${FILE}
    )

  spectre_test_timeout(TIMEOUT PYTHON 2)

  set(_PYTHONPATH "${SPECTRE_PYTHON_PREFIX_PARENT}")
  # Avoid "uninitialized variable" warnings
  if(DEFINED ENV{PYTHONPATH})
    set(_PYTHONPATH "${_PYTHONPATH}:$ENV{PYTHONPATH}")
  endif()
  set(_TEST_ENV_VARS "PYTHONPATH=${_PYTHONPATH}")
  if(BUILD_PYTHON_BINDINGS AND
      "${JEMALLOC_LIB_TYPE}" STREQUAL SHARED)
    list(APPEND
      _TEST_ENV_VARS
      "LD_PRELOAD=${JEMALLOC_LIBRARIES}"
      )
  endif()

  # The fail regular expression is what Python.unittest returns when no
  # tests are found to be run. We treat this as a test failure.
  set_tests_properties(
    "\"${TEST_NAME}\""
    PROPERTIES
    FAIL_REGULAR_EXPRESSION "Ran 0 test"
    TIMEOUT ${TIMEOUT}
    LABELS "${TAGS};Python"
    ENVIRONMENT "${_TEST_ENV_VARS}"
    )
  # check if this is a unit test, and if so add it to the dependencies
  foreach(LABEL ${TAGS})
    string(TOLOWER "${LABEL}" LOWER_LABEL)
    string(TOLOWER "${PY_MODULE_DEPENDENCY}" LOWER_DEP)
    if("${LOWER_LABEL}" STREQUAL "unit"
        AND NOT "${LOWER_DEP}" STREQUAL "none")
      add_dependencies(unit-tests ${PY_MODULE_DEPENDENCY})
    endif()
  endforeach()
endfunction()

# Register a python test file that uses bindings with ctest.
# - TEST_NAME    The name of the test,
#                e.g. "Unit.DataStructures.Python.DataVector"
#
# - FILE         The file to add, e.g. Test_DataVector.py
#
# - TAGS         A semicolon separated list of labels for the test,
#                e.g. "Unit;DataStructures;Python"
# - PY_MODULE_DEPENDENCY
#                The python module that this test depends on
#                Set to None if there is no python module dependency
function(SPECTRE_ADD_PYTHON_BINDINGS_TEST TEST_NAME FILE TAGS
    PY_MODULE_DEPENDENCY)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
  spectre_add_python_test(${TEST_NAME} ${FILE} "${TAGS}" ${PY_MODULE_DEPENDENCY})
endfunction()
