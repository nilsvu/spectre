# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Xcts)

set(LIBRARY_SOURCES
  Actions/LapseAtOrigin.cpp
  Equations.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE ErrorHandling
  INTERFACE Poisson
  )
