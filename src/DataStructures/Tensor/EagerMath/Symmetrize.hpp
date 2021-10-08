// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

template <typename DataType, size_t Dim, typename Frame>
void symmetrize(const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
                const tnsr::ij<DataType, Dim, Frame>& tensor) {
  for (size_t i = 0; i < Dim; ++i) {
    result->get(i, i) = tensor.get(i, i);
    for (size_t j = 0; j < i; ++j) {
      result->get(i, j) = 0.5 * (tensor.get(i, j) + tensor.get(j, i));
    }
  }
}
