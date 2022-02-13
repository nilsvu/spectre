// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/CompressedVector.h>
#include <pup.h>

namespace PUP {
/// @{
/// Serialization of blaze::CompressedVector
template <typename T, bool TF, typename Tag>
void pup(er& p, blaze::CompressedVector<T, TF, Tag>& t) {
  size_t size = t.size();
  p | size;
  size_t num_non_zeros = t.nonZeros();
  p | num_non_zeros;
  size_t index;
  if (p.isUnpacking()) {
    t.resize(size);
    for (size_t i = 0; i < num_non_zeros; ++i) {
      p | index;
      p | t[index];
    }
  } else {
    for (auto it = t.begin(); it != t.end(); ++it) {
      index = it->index();
      p | index;
      p | it->value();
    }
  }
}
template <typename T, bool TF, typename Tag>
void operator|(er& p, blaze::CompressedVector<T, TF, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP
