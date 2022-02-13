// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/CompressedMatrix.h>

#include "Options/ParseOptions.hpp"

template <typename Type, bool SO, typename Tag>
struct Options::create_from_yaml<blaze::CompressedMatrix<Type, SO, Tag>> {
  template <typename Metavariables>
  static blaze::CompressedMatrix<Type, SO, Tag> create(
      const Options::Option& options) {
    const auto data = options.parse_as<std::vector<std::vector<Type>>>();
    const size_t num_rows = data.size();
    size_t num_cols = 0;
    if (num_rows > 0) {
      num_cols = data[0].size();
    }
    blaze::CompressedMatrix<Type, SO, Tag> result(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; i++) {
      const auto& row = gsl::at(data, i);
      if (row.size() != num_cols) {
        PARSE_ERROR(options.context(),
                    "All matrix columns must have the same size.");
      }
      for (size_t j = 0; j < num_cols; j++) {
        if (gsl::at(row, j) != 0.) {
          result(i, j) = gsl::at(row, j);
        }
      }
    }
    return result;
  }
};
