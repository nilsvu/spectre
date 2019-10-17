# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PuncturesAnalyticData)

set(LIBRARY_SOURCES
  MultiplePunctures.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Utilities
  )
