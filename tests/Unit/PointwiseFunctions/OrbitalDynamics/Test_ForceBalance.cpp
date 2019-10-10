// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/OrbitalDynamics/ForceBalance.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

auto force_balance(const double eccentricity, const DataVector& position,
                   const DataVector& conformal_factor_pow_4,
                   const DataVector& lapse_square,
                   const DataVector shift_square, const DataVector& shift_y,
                   const DataVector& dx_conformal_factor_pow_4,
                   const DataVector& dx_lapse_square,
                   const DataVector& dx_shift_square,
                   const DataVector& dx_shift_y, const double center_of_mass,
                   const double angular_velocity) {
  return orbital::ForceBalance{
      eccentricity,    position,  conformal_factor_pow_4,    lapse_square,
      shift_square,    shift_y,   dx_conformal_factor_pow_4, dx_lapse_square,
      dx_shift_square, dx_shift_y}({{center_of_mass, angular_velocity}});
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.OrbitalDynamics.ForceBalance",
                  "[Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/OrbitalDynamics"};
  pypp::check_with_random_values<12>(&force_balance, "ForceBalance",
                                     "force_balance",
                                     {{{0., 0.5},
                                       {-1., 1.},
                                       {0.5, 1.5},
                                       {0.5, 1.5},
                                       {0., 0.5},
                                       {-0.5, 0.5},
                                       {-0.5, 0.5},
                                       {-0.5, 0.5},
                                       {-0.5, 0.5},
                                       {-0.5, 0.5},
                                       {-1., 1.},
                                       {0.1, 1.}}},
                                     DataVector{2, 0.});
}
