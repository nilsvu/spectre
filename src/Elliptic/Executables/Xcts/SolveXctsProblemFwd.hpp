// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BlackHoleBinary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Zero.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/BoundaryConditions/Xcts/ApparentHorizon.hpp"
#include "PointwiseFunctions/BoundaryConditions/Xcts/Binary.hpp"

/// \cond
namespace Xcts {
template <Equations EnabledEquations, Geometry ConformalGeometry>
struct FirstOrderSystem;
namespace Solutions {
struct ConstantDensityStar;
}  // namespace Solutions
}  // namespace Xcts

template <typename System, typename Background, typename BoundaryConditions,
          typename InitialGuess>
struct Metavariables;
/// \endcond
