// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

template <typename DataType>
void shift_strain(const gsl::not_null<tnsr::ii<DataType, 3>*> result,
                  const tnsr::iJ<DataType, 3>& deriv_shift,
                  const tnsr::ii<DataType, 3>& metric,
                  const tnsr::ijj<DataType, 3>& deriv_metric,
                  const tnsr::ijj<DataType, 3>& christoffel_first_kind,
                  const tnsr::I<DataType, 3>& shift) noexcept {
  auto deriv_shift_lowered =
      make_with_value<tnsr::ij<DataType, 3>>(deriv_shift, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_shift_lowered.get(i, j) +=
            metric.get(j, k) * deriv_shift.get(i, k) +
            shift.get(k) * deriv_metric.get(i, j, k);
      }
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      result->get(i, j) =
          0.5 * (deriv_shift_lowered.get(i, j) + deriv_shift_lowered.get(j, i));
      for (size_t k = 0; k < 3; ++k) {
        result->get(i, j) -=
            christoffel_first_kind.get(k, i, j) * shift.get(k);
      }
    }
  }
}

/*!
 * \brief The longitudinal operator, or vector gradient, \f$(L\beta)^{ij}\f$
 *
 * Computes the longitudinal operator
 *
 * \f{equation}
 * (L\beta)^{ij} = \nabla^i \beta^j + \nabla^j \beta^i -
 * \frac{2}{3}\gamma^{ij}\nabla_k\beta^k
 * \f}
 *
 * of a vector field \f$\beta^i\f$, where \f$\nabla\f$ denotes the covariant
 * derivative w.r.t. the metric \f$\gamma\f$ (see e.g. Eq. (3.50) in
 * \cite BaumgarteShapiro). Note that in the XCTS equations the longitudinal
 * operator is typically applied to conformal quantities and w.r.t. the
 * conformal metric \f$\bar{\gamma}\f$.
 *
 * In terms of the symmetric "strain" quantity
 * \f$B_{ij}=\nabla_{(i}\gamma_{j)k}\beta^k\f$ the longitudinal operator is:
 *
 * \f{equation}
 * (L\beta)^{ij} = 2\left(\gamma^{ik}\gamma^{jl} -
 * \frac{1}{3} \gamma^{jk}\gamma^{kl}\right) B_{kl}
 * \f}
 */
template <typename DataType>
void longitudinal_operator(gsl::not_null<tnsr::II<DataType, 3>*> result,
                           const tnsr::ii<DataType, 3>& strain,
                           const tnsr::II<DataType, 3>& inv_metric) noexcept;

/*!
 * \brief The conformal longitudinal operator \f$(L\beta)^{ij}\f$ on a flat
 * conformal metric in Cartesian coordinates \f$\gamma_{ij}=\delta_{ij}\f$
 *
 * \see `Xcts::longitudinal_operator`
 */
template <typename DataType>
void longitudinal_operator_flat_cartesian(
    gsl::not_null<tnsr::II<DataType, 3>*> result,
    const tnsr::ii<DataType, 3>& strain) noexcept;
}  // namespace Xcts
