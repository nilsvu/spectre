# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ElasticitySolutions)

set(LIBRARY_SOURCES
  BentBeam.cpp
  HalfSpaceMirror.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  PUBLIC Integration
  INTERFACE ErrorHandling
  INTERFACE Utilities
  INTERFACE ConstitutiveRelations
  PRIVATE GSL::gsl
  )
