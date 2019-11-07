# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(GrMhd)

set(LIBRARY "Test_AnalyticData")

set(LIBRARY_SOURCES
  Test_Protocols.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/"
  "${LIBRARY_SOURCES}"
  "DataStructures"
  )
