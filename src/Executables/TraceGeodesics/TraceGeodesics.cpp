// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Executables/TraceGeodesics/TraceGeodesics.hpp"

#include <vector>

#include "Parallel/CharmMain.tpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

using metavariables = ray_tracing::Metavariables;

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&register_factory_classes_with_charm<metavariables>,
       &EquationsOfState::register_derived_with_charm},
      {});
}
