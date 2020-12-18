# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Charm 6.10.2 EXACT REQUIRED
  COMPONENTS
  CommonLBs
  )

# Note that Charm++ provides a `charmc` script that wraps the compiler. We avoid
# that to reduce complexity in the build system. The `FindCharm` module provides
# imported targets that carry the include directories, libraries and compiler
# flags that the `charmc` script would otherwise set. Here's how we handle
# `charmc`'s remaining responsibilities:
# - `charmc` writes a tiny temporary `moduleinit$$.C` file with a few functions
#   that usually do nothing, compiles it and links it into every Charm module.
#   We could have `charmc` generate and _not_ clean up this file py passing the
#   `-save` option to `charmc`, e.g. in our `add_charm_module` CMake function.
#   However, the `charmc` script does not intend to expose this file so trying
#   to extract it can be very fragile. Until we actually need any of these
#   "moduleinit" functions to do something we just provide trivial
#   implementations for them in `src/Parallel/CharmModuleInit.cpp`. See this
#   upstream issue for details: https://github.com/UIUC-PPL/charm/issues/3210

add_interface_lib_headers(
  TARGET
  Charmxx::charmxx
  HEADERS
  charm++.h
  )
add_interface_lib_headers(
  TARGET
  Charmxx::pup
  HEADERS
  pup.h
  pup_stl.h
  )
set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Charmxx::charmxx Charmxx::pup
  )

get_filename_component(CHARM_BINDIR ${CHARM_COMPILER} DIRECTORY)
# In order to avoid problems when compiling in parallel we manually copy the
# charmrun script over, rather than having charmc do it for us.
configure_file(
    "${CHARM_BINDIR}/charmrun"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/charmrun" COPYONLY
)

include(SetupCharmModuleFunctions)

set(SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD ${CHARM_SMP})
if (${SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD})
  message(STATUS "Charm++ built with shared memory parallelism")
  add_definitions(-DSPECTRE_SHARED_MEMORY_PARALLELISM_BUILD)
else()
  message(STATUS "Charm++ NOT built with shared memory parallelism")
endif()

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Charm Version:  ${CHARM_VERSION}\n"
  "Charm SMP:  ${SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD}\n"
  )
