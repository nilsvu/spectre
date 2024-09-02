// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Executables/SelfForce/Scalar/SolveScalarSelfForce.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Elliptic/SubdomainPreconditioners/RegisterDerived.hpp"
#include "Parallel/CharmMain.tpp"
#include "ParallelAlgorithms/Amr/Actions/RegisterCallbacks.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &register_derived_classes_with_charm<
           Metavariables::solver::schwarz_smoother::subdomain_solver>,
       &elliptic::subdomain_preconditioners::register_derived_with_charm,
       &register_factory_classes_with_charm<Metavariables>,
       &::amr::register_callbacks<Metavariables,
                                  typename Metavariables::amr::element_array>},
      {});
}
