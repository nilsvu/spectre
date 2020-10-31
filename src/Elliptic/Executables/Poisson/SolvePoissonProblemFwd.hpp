// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/Systems/Poisson/Geometry.hpp"

/// \cond
namespace Poisson {
template <size_t Dim, Geometry BackgroundGeometry>
struct FirstOrderSystem;
namespace Solutions {
template <size_t Dim>
struct Lorentzian;
template <size_t Dim>
struct Moustache;
template <size_t Dim>
struct ProductOfSinusoids;
template <size_t Dim>
struct Zero;
}  // namespace Solutions
}  // namespace Poisson

template <typename System, typename Background, typename BoundaryConditions,
          typename InitialGuess>
struct Metavariables;
/// \endcond
