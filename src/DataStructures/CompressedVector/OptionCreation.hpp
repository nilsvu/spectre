// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/CompressedVector.h>

#include "Options/ParseOptions.hpp"
#include "Utilities/Gsl.hpp"

template <typename T, bool TF, typename Tag>
struct Options::create_from_yaml<blaze::CompressedVector<T, TF, Tag>> {
  template <typename Metavariables>
  static blaze::CompressedVector<T, TF, Tag> create(
      const Options::Option& options) {
    const auto data = options.parse_as<std::vector<T>>();
    blaze::CompressedVector<T, TF, Tag> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      if (gsl::at(data, i) != 0.) {
        result[i] = gsl::at(data, i);
      }
    }
    return result;
  }
};
