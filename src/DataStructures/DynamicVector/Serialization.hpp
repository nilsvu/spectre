// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/DynamicVector.h>
#include <pup.h>

namespace PUP {
/// @{
/// Serialization of blaze::DynamicVector
template <typename T, bool TF, typename Tag>
void pup(er& p, blaze::DynamicVector<T, TF, Tag>& t) {
  size_t size = t.size();
  p | size;
  if (p.isUnpacking()) {
    t.resize(size);
  }
  if (std::is_fundamental_v<T>) {
    PUParray(p, t.data(), size);
  } else {
    for (T& element : t) {
      p | element;
    }
  }
}
template <typename T, bool TF, typename Tag>
void operator|(er& p, blaze::DynamicVector<T, TF, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP
