// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/DynamicMatrix.h>

#include "Options/ParseOptions.hpp"

template <typename Type, bool SO, typename Alloc, typename Tag>
struct Options::create_from_yaml<blaze::DynamicMatrix<Type, SO, Alloc, Tag>> {
  template <typename Metavariables>
  static blaze::DynamicMatrix<Type, SO, Alloc, Tag> create(
      const Options::Option& options) {
    const auto data = options.parse_as<std::vector<std::vector<Type>>>();
    const size_t num_rows = data.size();
    size_t num_cols = 0;
    if (num_rows > 0) {
      num_cols = data[0].size();
    }
    blaze::DynamicMatrix<Type, SO, Alloc, Tag> result(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; i++) {
      const auto& row = gsl::at(data, i);
      if (row.size() != num_cols) {
        PARSE_ERROR(options.context(),
                    "All matrix columns must have the same size.");
      }
      for (size_t j = 0; j < num_cols; j++) {
        result(i, j) = gsl::at(row, j);
      }
    }
    return result;
  }
};
