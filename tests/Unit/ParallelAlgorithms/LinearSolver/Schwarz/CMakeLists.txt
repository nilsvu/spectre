# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelSchwarz")

# set(LIBRARY_SOURCES
#   Test_ElementActions.cpp
#   Test_ResidualMonitorActions.cpp
#   )

# add_test_library(
#   ${LIBRARY}
#   "ParallelAlgorithms/LinearSolver/Gmres"
#   "${LIBRARY_SOURCES}"
#   "Convergence;DataStructures;IO"
#   )

# add_dependencies(
#   ${LIBRARY}
#   module_ConstGlobalCache
#   )

# This code is adapted from Parallel/CMakeLists.txt

add_linear_solver_algorithm_test("DistributedSchwarzAlgorithm")
