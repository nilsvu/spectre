# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Burgers)
add_subdirectory(Elasticity)
add_subdirectory(GeneralRelativity)
add_subdirectory(GrMhd)
add_subdirectory(NewtonianEuler)
add_subdirectory(Poisson)
add_subdirectory(RadiationTransport)
add_subdirectory(RelativisticEuler)
add_subdirectory(WaveEquation)
add_subdirectory(Xcts)

set(LIBRARY "Test_AnalyticSolutions")

set(LIBRARY_SOURCES
  Test_Protocols.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/"
  "${LIBRARY_SOURCES}"
  "DataStructures"
  )
