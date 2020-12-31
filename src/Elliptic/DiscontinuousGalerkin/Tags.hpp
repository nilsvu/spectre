// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"

namespace elliptic::dg::Tags {

template <typename Tag, typename Dim, typename Frame>
using NormalDotDivAuxFlux =
    ::Tags::NormalDotFlux<::Tags::div<::Tags::Flux<Tag, Dim, Frame>>>;

}  // namespace elliptic::dg::Tags
