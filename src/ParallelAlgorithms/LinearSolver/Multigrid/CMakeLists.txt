# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelMultigrid)

set(LIBRARY_SOURCES
  InterMeshOperators.cpp
  MeshHierarchy.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  PRIVATE
  Spectral
  )
