// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

/// \cond
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
        result->get(i, j) -= christoffel_first_kind.get(k, i, j) * shift.get(k);
      }
    }
  }
}

template <typename DataType>
void longitudinal_operator(const gsl::not_null<tnsr::II<DataType, 3>*> result,
                           const tnsr::ii<DataType, 3>& strain,
                           const tnsr::II<DataType, 3>& inv_metric) noexcept {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      // Unroll first iteration of the loop over `k` to avoid filling the result
      // with zero initially. This first assignment is the k=0, l=0 iteration:
      result->get(i, j) =
          (2. * inv_metric.get(i, 0) * inv_metric.get(j, 0) -
           2. / 3. * inv_metric.get(i, j) * get<0, 0>(inv_metric)) *
          get<0, 0>(strain);
      // These are the remaining contributions of the k=0 iteration:
      for (size_t l = 1; l < 3; ++l) {
        result->get(i, j) +=
            (2. * inv_metric.get(i, 0) * inv_metric.get(j, l) -
             2. / 3. * inv_metric.get(i, j) * inv_metric.get(0, l)) *
            strain.get(0, l);
      }
      // This is the loop from which the k=0 iteration is unrolled above:
      for (size_t k = 1; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          result->get(i, j) +=
              (2. * inv_metric.get(i, k) * inv_metric.get(j, l) -
               2. / 3. * inv_metric.get(i, j) * inv_metric.get(k, l)) *
              strain.get(k, l);
        }
      }
    }
  }
}

template <typename DataType>
void longitudinal_operator_flat_cartesian(
    const gsl::not_null<tnsr::II<DataType, 3>*> result,
    const tnsr::ii<DataType, 3>& strain) noexcept {
  // Compute trace term in 2-2 component of the result
  get<2, 2>(*result) = get<0, 0>(strain);
  for (size_t d = 1; d < 3; ++d) {
    get<2, 2>(*result) += strain.get(d, d);
  }
  get<2, 2>(*result) *= -2. / 3.;
  for (size_t i = 0; i < 3; ++i) {
    // Copy trace term to other diagonal components and complete diagonal
    // components with non-trace contribution
    result->get(i, i) = get<2, 2>(*result) + 2. * strain.get(i, i);
    // Compute off-diagonal contributions
    for (size_t j = 0; j < i; ++j) {
      result->get(i, j) = 2. * strain.get(i, j);
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                   \
  template void shift_strain(                                  \
      gsl::not_null<tnsr::ii<DTYPE(data), 3>*> result,         \
      const tnsr::iJ<DTYPE(data), 3>& deriv_shift,             \
      const tnsr::ii<DTYPE(data), 3>& metric,                  \
      const tnsr::ijj<DTYPE(data), 3>& deriv_metric,           \
      const tnsr::ijj<DTYPE(data), 3>& christoffel_first_kind, \
      const tnsr::I<DTYPE(data), 3>& shift) noexcept;          \
  template void longitudinal_operator(                         \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,         \
      const tnsr::ii<DTYPE(data), 3>& strain,                  \
      const tnsr::II<DTYPE(data), 3>& inv_metric) noexcept;    \
  template void longitudinal_operator_flat_cartesian(          \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,         \
      const tnsr::ii<DTYPE(data), 3>& strain) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond

}  // namespace Xcts
