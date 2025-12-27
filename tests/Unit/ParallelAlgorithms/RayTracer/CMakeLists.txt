# Distributed under the MIT License.
# See LICENSE.txt for details.

if (NOT TARGET Kokkos::kokkos)
  return()
endif()

set(LIBRARY "Test_RayTracer")

set(LIBRARY_SOURCES
  Test_RayTracer.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  RayTracer
  )

add_subdirectory(BackgroundSpacetimes)
add_subdirectory(Python)
add_subdirectory(RaySources)
