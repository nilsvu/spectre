# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Python)

set(LIBRARY LinearSolver)

set(LIBRARY_SOURCES
  Gmres.cpp
  Lapack.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  Boost::boost
  Convergence
  DataStructures
  Options

  PRIVATE
  Lapack
  )
