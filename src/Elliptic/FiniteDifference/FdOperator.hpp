// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace elliptic::fd {

template <size_t Dim>
void apply_operator(
    const gsl::not_null<DataVector*> result, const DataVector& u,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian) {
  const size_t num_points = mesh.number_of_grid_points();
  set_number_of_grid_points(result, num_points);
  *result = 0.;

  // Fourth-order finite difference stencil
  // -1/12  4/3  -5/2  4/3  -1/12
  const std::array<double, 3> stencil{{-5. / 2., 4. / 3., -1. / 12.}};
  for (size_t d = 0; d < Dim; ++d) {
    const size_t num_points_d = mesh.extents(d);
    // TODO: generalize to non-diagonal and non-constant Jacobian
    const double jacobian_factor =
        square(inv_jacobian.get(d, d)[0] * mesh.extents(d) / 2.);
    for (StripeIterator si(mesh.extents(), d); si; ++si) {
      const auto storage_index = [&si](const size_t i) {
        return i * si.stride() + si.offset();
      };
      for (size_t i = 0; i < num_points_d; ++i) {
        auto& result_i = (*result)[storage_index(i)];
        result_i += u[storage_index(i)] * stencil[0] * jacobian_factor;
        if (i > 0) {
          result_i += u[storage_index(i - 1)] * stencil[1] * jacobian_factor;
        }
        if (i + 1 < num_points_d) {
          result_i += u[storage_index(i + 1)] * stencil[1] * jacobian_factor;
        }
        if (i > 1) {
          result_i += u[storage_index(i - 2)] * stencil[2] * jacobian_factor;
        }
        if (i + 2 < num_points_d) {
          result_i += u[storage_index(i + 2)] * stencil[2] * jacobian_factor;
        }
      }
    }
  }
}

}  // namespace elliptic::fd
