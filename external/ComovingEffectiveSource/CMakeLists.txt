# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ComovingEffectiveSource)

add_library(${LIBRARY} 
  kerr-circular-comoving.c
  Transformation.cpp)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GSL::gsl
  GSL::gslcblas
  )

target_include_directories(${LIBRARY} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
