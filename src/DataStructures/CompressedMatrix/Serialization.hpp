// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/CompressedMatrix.h>
#include <pup.h>

namespace PUP {
/// @{
/// Serialization of blaze::DynamicMatrix
template <typename Type, bool SO, typename Tag>
void pup(er& p, blaze::CompressedMatrix<Type, SO, Tag>& t) {
  size_t rows = t.rows();
  size_t columns = t.columns();
  p | rows;
  p | columns;
  const size_t first_dimension = (SO == blaze::rowMajor) ? rows : columns;
  size_t num_non_zeros;
  size_t index;
  if (p.isUnpacking()) {
    t.resize(rows, columns);
    for (size_t i = 0; i < first_dimension; ++i) {
      p | num_non_zeros;
      for (size_t j = 0; j < num_non_zeros; ++j) {
        p | index;
        if constexpr (SO == blaze::rowMajor) {
          p | t(i, index);
        } else {
          p | t(index, i);
        }
      }
    }
  } else {
    for (size_t i = 0; i < first_dimension; ++i) {
      num_non_zeros = t.nonZeros(i);
      p | num_non_zeros;
      for (auto it = t.begin(i); it != t.end(i); ++it) {
        index = it->index();
        p | index;
        p | it->value();
      }
    }
  }
}
template <typename Type, bool SO, typename Tag>
void operator|(er& p, blaze::CompressedMatrix<Type, SO, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP
