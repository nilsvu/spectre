# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY KokkosTest)

add_spectre_executable(${LIBRARY} Test.cpp)

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  DataStructures
  SpectreKokkos
  )
