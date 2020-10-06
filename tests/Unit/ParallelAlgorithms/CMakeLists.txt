# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelAlgorithms")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/"
  "${LIBRARY_SOURCES}"
  "DataStructures;ParallelAlgorithms;Utilities"
  )

add_subdirectory(Actions)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Events)
add_subdirectory(EventsAndTriggers)
add_subdirectory(Initialization)
add_subdirectory(LinearSolver)
