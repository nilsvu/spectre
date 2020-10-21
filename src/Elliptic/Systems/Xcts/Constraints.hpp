// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

template <typename DataType, size_t Dim, typename Frame>
void hamiltonian_constraint(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& deriv_spatial_metric,
    const tnsr::ij<DataVector, Dim, Frame>& extrinsic_curvature) noexcept {
  const auto inv_spatial_metric = determinant_and_inverse(spatial_metric);
  const auto christoffel_2nd_kind = raise_or_lower_first_index(
      gr::christoffel_first_kind(deriv_spatial_metric), inv_spatial_metric);
  // TODO: deriv_christoffel_2nd_kind
  const auto ricci_tensor =
      gr::ricci_tensor(christoffel_2nd_kind, deriv_christoffel_2nd_kind);
  // TODO: ricci_scalar
  // TODO: extrinsic_curvature_trace
  // TODO: extrinsic_curvature_square
  get(*result) = ricci_scalar + square(get(extrinsic_curvature_trace)) -
                 extrinsic_curvature_square;
}

// template <typename DataType>
// Scalar<DataType> hamiltonian_constraint() noexcept {
//   auto result = make_with_value<Scalar<DataType>>(
//       , std::numeric_limits<double>::signaling_NaN());
//   hamiltonian_constraint(make_not_null(&result), );
//   return result;
// }

}  // namespace Xcts
