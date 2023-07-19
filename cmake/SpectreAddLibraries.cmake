# Distributed under the MIT License.
# See LICENSE.txt for details.

add_custom_target(libs)

# Wraps CMake's `add_library` to handle PCH, compiler flags, etc.
#
# Arguments:
# - NO_PCH (optional): Don't use PCH (precompiled headers). Use this option to
#   exclude PCH from libraries that the PCH depend on.
#
# All additional arguments are forwarded to `add_library`.
function(ADD_SPECTRE_LIBRARY LIBRARY_NAME)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "NO_PCH" "" "")
  list(REMOVE_ITEM ARGN NO_PCH)

  add_library(${LIBRARY_NAME} ${ARGN})
  add_dependencies(libs ${LIBRARY_NAME})

  get_target_property(
    LIBRARY_IS_IMPORTED
    ${LIBRARY_NAME}
    IMPORTED
    )
  get_target_property(
    LIBRARY_TYPE
    ${LIBRARY_NAME}
    TYPE
    )

  # We need to link custom allocators before we link anything else so that
  # any third-party libraries, which generally should all be built as shared
  # libraries, use the allocator that we use. Unfortunately, how exactly
  # CMake decides on the linking order is not clear when using
  # INTERFACE_LINK_LIBRARIES and targets. To this end, we set a global
  # property SPECTRE_ALLOCATOR_LIBRARY that contains the link flag to link
  # to the memory allocator. By linking to the allocator library first
  # explicitly in target_link_libraries CMake correctly places the allocator
  # library as the first entry in the link libraries. We also link to the
  # SpectreAllocator target to pull in any additional allocator-related
  # flags, such as include directories.
  get_property(
    SPECTRE_ALLOCATOR_LIBRARY
    GLOBAL
    PROPERTY SPECTRE_ALLOCATOR_LIBRARY
    )
  if (NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
    target_link_libraries(${LIBRARY_NAME}
      PUBLIC
      ${SPECTRE_ALLOCATOR_LIBRARY}
      SpectreAllocator
      )

    set_target_properties(
      ${LIBRARY_NAME}
      PROPERTIES
      RULE_LAUNCH_LINK "${CMAKE_BINARY_DIR}/tmp/WrapLibraryLinker.sh"
      LINK_DEPENDS "${CMAKE_BINARY_DIR}/tmp/WrapLibraryLinker.sh"
      )
  endif (NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
  if (NOT "${LIBRARY_NAME}" MATCHES "^${SPECTRE_PCH}"
      AND NOT ${LIBRARY_IS_IMPORTED}
      AND NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY
      AND NOT "${ARG_NO_PCH}"
      AND TARGET ${SPECTRE_PCH})
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${SPECTRE_PCH})
    add_dependencies(${LIBRARY_NAME} ${SPECTRE_PCH_DEP})
    set_source_files_properties(
        ${ARGN}
        OBJECT_DEPENDS "${SPECTRE_PCH_PATH}"
        )
  endif()
  if (${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
    target_link_libraries(
      ${LIBRARY_NAME}
      INTERFACE
      SpectreFlags
      )
  else()
    target_link_libraries(
      ${LIBRARY_NAME}
      PUBLIC
      SpectreFlags
      )
    set_property(
      TARGET ${LIBRARY_NAME}
      PROPERTY FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
      )
  endif()
  if (BUILD_SHARED_LIBS
      AND NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
    install(TARGETS ${LIBRARY_NAME} OPTIONAL
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endfunction()
