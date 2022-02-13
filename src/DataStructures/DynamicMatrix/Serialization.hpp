// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/DynamicMatrix.h>
#include <pup.h>

namespace PUP {
/// @{
/// Serialization of blaze::DynamicMatrix
template <typename Type, bool SO, typename Alloc, typename Tag>
void pup(er& p, blaze::DynamicMatrix<Type, SO, Alloc, Tag>& t) {
  size_t rows = t.rows();
  size_t columns = t.columns();
  p | rows;
  p | columns;
  if (p.isUnpacking()) {
    t.resize(rows, columns);
  }
  size_t spacing = t.spacing();
  size_t data_size = spacing * (SO == blaze::rowMajor ? rows : columns);
  if (std::is_fundamental_v<Type>) {
    PUParray(p, t.data(), data_size);
  } else {
    for (size_t i = 0; i < data_size; ++i) {
      p | t.data()[i];
    }
  }
}
template <typename Type, bool SO, typename Alloc, typename Tag>
void operator|(er& p, blaze::DynamicMatrix<Type, SO, Alloc, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP
