# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Eigen3)

if (NOT Eigen3_FOUND)
  if (NOT SPECTRE_FETCH_MISSING_DEPS)
    message(FATAL_ERROR "Could not find Eigen. If you want to fetch "
      "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
  endif()
  message(STATUS "Fetching Eigen")
  include(FetchContent)
  FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    ${SPECTRE_FETCHCONTENT_BASE_ARGS}
  )
  set(EIGEN_BUILD_TESTING OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  set(EIGEN_BUILD_DOC OFF)
  set(EIGEN_DONT_PARALLELIZE ON)
  FetchContent_MakeAvailable(Eigen3)
else()
  target_compile_definitions(
    Eigen3::Eigen
    INTERFACE
    EIGEN_DONT_PARALLELIZE
  )
endif()
