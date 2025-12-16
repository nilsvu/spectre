// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

template <typename DataType, typename Index, size_t NumVectors>
void gram_schmidt_orthonormalize(
    const std::array<
        gsl::not_null<Tensor<DataType, Symmetry<1>, index_list<Index>>*>,
        NumVectors>& basis,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) {
  {  // Normalize the first vector
    auto& first = *basis[0];
    const auto norm = magnitude(first, metric);
    for (size_t k = 0; k < 4; ++k) {
      first.get(k) /= get(norm);
    }
  }
  // Orthogonalize the remaining vectors
  for (size_t i = 1; i < basis.size(); ++i) {
    auto& v = *basis[i];
    for (size_t j = 0; j < i; ++j) {
      auto& w = *basis[j];
      const auto projection =
          get(dot_product(v, w, metric)) / get(dot_product(w, w, metric));
      for (size_t k = 0; k < 4; ++k) {
        v.get(k) -= projection * w.get(k);
      }
      const auto norm = magnitude(v, metric);
      for (size_t k = 0; k < 4; ++k) {
        v.get(k) /= get(norm);
      }
    }
  }
}
