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
namespace AnalyticData {
struct Mirror;
}  // namespace AnalyticData
namespace BoundaryConditions {
struct LaserBeam;
}  // namespace BoundaryConditions
}  // namespace Elasticity
namespace elliptic {
template <typename SolutionType>
struct InitialGuessFromSolution;
}  // namespace elliptic

template <typename System, typename Background, typename BoundaryConditions,
          typename InitialGuess>
struct Metavariables;
/// \endcond
