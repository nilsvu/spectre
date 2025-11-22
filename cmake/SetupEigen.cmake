# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Eigen3)

target_compile_definitions(
  Eigen3::Eigen
  INTERFACE
  EIGEN_DONT_PARALLELIZE
)
