# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_OrbitalDynamics")

set(LIBRARY_SOURCES
  Test_ForceBalance.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/OrbitalDynamics/"
  "${LIBRARY_SOURCES}"
  "Test_Pypp;OrbitalDynamics"
  )
