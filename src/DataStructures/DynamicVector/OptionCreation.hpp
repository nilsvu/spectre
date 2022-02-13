// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/DynamicVector.h>

#include "Options/ParseOptions.hpp"

template <typename T, bool TF, typename Tag>
struct Options::create_from_yaml<blaze::DynamicVector<T, TF, Tag>> {
  template <typename Metavariables>
  static blaze::DynamicVector<T, TF, Tag> create(
      const Options::Option& options) {
    const auto data = options.parse_as<std::vector<T>>();
    blaze::DynamicVector<T, TF, Tag> result(data.size());
    std::copy(std::begin(data), std::end(data), result.begin());
    return result;
  }
};
