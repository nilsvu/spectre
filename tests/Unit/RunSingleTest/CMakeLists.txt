# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "RunSingleTest")

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ${EXECUTABLE}.cpp
  # The following lines list the source files with tests in them to be
  # compiled into the executable.
  ../Elliptic/DiscontinuousGalerkin/Test_StrongSecondOrderInternalPenalty.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  DataStructures
  LinearOperators
  Domain
  DomainCreators
  Spectral
  Utilities
  Informer
  ErrorHandling
  ${SPECTRE_LIBRARIES}
  )

set_target_properties(
  ${EXECUTABLE}
  PROPERTIES
  LINK_FLAGS "-nomain -nomain-module"
  )
