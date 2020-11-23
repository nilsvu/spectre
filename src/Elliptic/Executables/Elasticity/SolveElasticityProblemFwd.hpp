// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace Elasticity {
template <size_t Dim>
struct FirstOrderSystem;
namespace Solutions {
struct BentBeam;
struct HalfSpaceMirror;
template <size_t Dim>
struct Zero;
}  // namespace Solutions
}  // namespace Elasticity

template <typename System, typename Background, typename BoundaryConditions,
          typename InitialGuess>
struct Metavariables;
/// \endcond
