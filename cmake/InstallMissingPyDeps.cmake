# Distributed under the MIT License.
# See LICENSE.txt for details.

# Optionally install missing Python dependencies to CMAKE_BINARY_DIR

option(INSTALL_PY_DEPS
  "Install missing Python dependencies in the build directory" OFF)

if (NOT INSTALL_PY_DEPS)
  return()
endif()

find_package(Python)

# The packages are installed into this directory within the build dir:
set(SPECTRE_PYTHON_SITE_PACKAGES "${CMAKE_BINARY_DIR}/lib/\
python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")

message(STATUS "Install missing Python packages to: \
${SPECTRE_PYTHON_SITE_PACKAGES}")

# Extract 'install_requires' section from setup.cfg
execute_process(COMMAND ${Python_EXECUTABLE} -c "
import configparser
config = configparser.ConfigParser()
config.read('${CMAKE_SOURCE_DIR}/src/PythonBindings/setup.cfg')
with open('${CMAKE_BINARY_DIR}/tmp/requirements.txt', 'w') as open_file:
  open_file.write(config['options']['install_requires'])
")

# Install the packages with pip
execute_process(COMMAND
  ${CMAKE_COMMAND} -E env
  PYTHONPATH=${SPECTRE_PYTHON_SITE_PACKAGES}
  ${Python_EXECUTABLE} -m pip install --disable-pip-version-check
  --prefix ${CMAKE_BINARY_DIR} -r ${CMAKE_BINARY_DIR}/tmp/requirements.txt)
