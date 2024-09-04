// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Angular.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Sommerfeld.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::BoundaryConditions {

using standard_boundary_conditions = tmpl::list<Angular, Sommerfeld>;

}
