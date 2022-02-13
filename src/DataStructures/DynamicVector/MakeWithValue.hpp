// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/DynamicVector.h>

#include "Utilities/MakeWithValue.hpp"

namespace MakeWithValueImpls {
template <typename T, bool TF, typename Tag>
struct NumberOfPoints<blaze::DynamicVector<T, TF, Tag>> {
  static SPECTRE_ALWAYS_INLINE size_t
  apply(const blaze::DynamicVector<T, TF, Tag>& input) {
    return input.size();
  }
};

template <typename T, bool TF, typename Tag>
struct MakeWithSize<blaze::DynamicVector<T, TF, Tag>> {
  static SPECTRE_ALWAYS_INLINE blaze::DynamicVector<T, TF, Tag> apply(
      const size_t size, const T& value) {
    return blaze::DynamicVector<T, TF, Tag>(size, value);
  }
};
}  // namespace MakeWithValueImpls
