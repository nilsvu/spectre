// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/CompressedMatrix.h>
#include <pup.h>

namespace PUP {

template <typename Type, bool SO>
void pup(er& p, blaze::CompressedMatrix<Type, SO>& matrix) noexcept {
  size_t rows = matrix.rows();
  size_t columns = matrix.columns();
  size_t num_nonzeros = matrix.nonZero();
  p | rows;
  p | columns;
  p | num_nonzeros;
  if (p.isUnpacking()) {
    matrix.resize(rows, columns);
    matrix.reset();
    matrix.reserve(num_nonzeros);
  }
  for (size_t i = 0; i < (SO == blaze::columnMajor ? columns : rows); ++i) {
    for (size_t j = 0; j < (SO == blaze::columnMajor ? rows : columns); ++j) {
      // TODO
      if (p.isUnpacking()) {
        matrix.append();
      }
    }
  }
}

/// \ingroup ParallelGroup
/// Serialization of std::optional for Charm++
template <typename Type, bool SO>
void operator|(er& p, blaze::CompressedMatrix<Type, SO>& matrix) noexcept {
  pup(p, matrix);
}

}  // namespace PUP
