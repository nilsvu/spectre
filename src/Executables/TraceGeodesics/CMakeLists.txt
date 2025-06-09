# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE TraceGeodesics3D)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  TraceGeodesics.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Charmxx::main
  DataStructures
  Hydro
  Options
  Parallel
  RayTracer
  Utilities
  )
