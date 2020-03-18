# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Boost 1.60.0 REQUIRED COMPONENTS program_options)

message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Boost Version:  ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}\n"
  )

add_library(Boost INTERFACE IMPORTED)
set_property(TARGET Boost
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
set_property(TARGET Boost PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
