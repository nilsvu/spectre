// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace elliptic::dg {
template <size_t Dim>
Mesh<Dim> oversample(const Mesh<Dim>& mesh,
                     const size_t oversample_points) noexcept {
  std::array<size_t, Dim> oversampled_extents = mesh.extents().indices();
  if (oversample_points > 0) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(oversampled_extents, d) += oversample_points;
    }
  }
  return Mesh<Dim>{oversampled_extents, mesh.basis(), mesh.quadrature()};
}
}  // namespace elliptic::dg
