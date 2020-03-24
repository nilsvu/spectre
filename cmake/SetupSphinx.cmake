# Distributed under the MIT License.
# See LICENSE.txt for details.


# While we cannot fully support Sphinx+Breathe yet because of some
# upstream bugs in Sphinx (Sphinx issues #7367 and #7368) we have
# support for using Sphinx once the issues are resolved. We have
# a SpECTRE issue #2138 for tracking Sphinx progress.
#
# Some notes:
# - The issues in Sphinx will (hopefully) be resolved on Sphinx v3.0
#   timescales, though one of the issues is a C++ parsing issue and
#   parsing C++ is notoriously difficult.
# - Currently one must manually run make doc-xml before running Sphinx
#   in order to get the Doxygen XML output. We will want to automate
#   this in the future with true tracking of files needed for building
#   Doxygen and Sphinx. Doing so will reduce the time it takes to rebuild
#   documentation.
# - Breathe's XML parser is horribly slow. See SpECTRE issue #2138
# - We will want to auto-generate all the Group documentation and
#   namespace documentation. For Doxygen groups this should be easy since
#   we can just simply parse `GroupDefs.hpp` and generate an RST file from
#   that. The namespaces will be more difficult, but one option would be
#   to read and parse Doxygen's XML using Beautifulsoup4 or lxml. There is
#   a package, Exhale, that should generate both of these for us but I got
#   errors about "multiple references for..." that as far as I can tell are
#   related to overloads (and maybe template specializations). Regardless,
#   at least the groups will be easy to parse, and if parsing namespaces
#   from XML isn't too bad we can also parse the groups that way. We could
#   even write a Sphinx extension for ourselves. My only worry with the
#   Sphinx extension is that I don't know how much maintenance that'll
#   require to keep working as Sphinx changes, while a simple python script
#   that generates RST files would fairly be easy to maintain independent
#   of what Sphinx does.
option(USE_SPHINX
  "When enabled, find and set up Sphinx+Breathe for documentation"
  OFF)

if (DOXYGEN_FOUND AND USE_SPHINX)
  find_package(Sphinx REQUIRED)
  find_package(Breathe REQUIRED)

  configure_file(
    ${CMAKE_SOURCE_DIR}/docs/conf.py
    ${CMAKE_BINARY_DIR}/docs/conf.py
    @ONLY
    )

  configure_file(
    ${CMAKE_SOURCE_DIR}/docs/index.rst
    ${CMAKE_BINARY_DIR}/docs/index.rst
    )

  set(SPHINX_SOURCE ${CMAKE_BINARY_DIR}/docs)
  set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/docs/sphinx)
  set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

  add_custom_target(Sphinx ALL
    COMMAND
    ${SPHINX_EXECUTABLE} -b html
    ${SPHINX_SOURCE} ${SPHINX_BUILD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Sphinx"
    )
endif(DOXYGEN_FOUND AND USE_SPHINX)
