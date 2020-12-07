# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_PETSC "Try to find PETSc" ON)

if(USE_PETSC)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(Petsc REQUIRED IMPORTED_TARGET petsc)
endif()
