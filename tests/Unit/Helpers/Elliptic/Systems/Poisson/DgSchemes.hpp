// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
struct Matrix;
template <size_t Dim>
struct DomainCreator;
/// \endcond

namespace TestHelpers {
namespace Poisson {

template <size_t Dim>
Matrix strong_first_order_dg_operator_matrix(
    const DomainCreator<Dim>& domain_creator);

}  // namespace Poisson
}  // namespace TestHelpers
