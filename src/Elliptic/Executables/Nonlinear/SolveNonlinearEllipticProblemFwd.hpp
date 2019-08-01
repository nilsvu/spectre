// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace Poisson {
template <size_t Dim>
struct FirstOrderCorrectionSystem;
namespace Solutions {
template <size_t Dim>
struct ProductOfSinusoids;
}  // namespace Solutions
}  // namespace Poisson

template <typename System, typename InitialGuess>
struct Metavariables;
