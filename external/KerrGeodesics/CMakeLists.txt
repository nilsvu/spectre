# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(FFTW)

set(LIBRARY KerrGeodesics)

add_library(${LIBRARY} korb.c)

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  FFTW::FFTW
  GSL::gsl
  )

target_include_directories(${LIBRARY} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
