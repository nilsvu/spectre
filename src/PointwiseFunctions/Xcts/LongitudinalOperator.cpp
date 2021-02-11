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
void longitudinal_operator(const gsl::not_null<tnsr::II<DataType, 3>*> result,
                           const tnsr::ii<DataType, 3>& strain,
                           const tnsr::II<DataType, 3>& inv_metric) noexcept {
  std::fill(result->begin(), result->end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          auto projection =
              2. * (inv_metric.get(i, k) * inv_metric.get(j, l) -
                    inv_metric.get(i, j) * inv_metric.get(k, l) / 3.);
          result->get(i, j) += projection * strain.get(k, l);
        }
      }
    }
  }
}

template <typename DataType>
void longitudinal_operator_flat_cartesian(
    const gsl::not_null<tnsr::II<DataType, 3>*> result,
    const tnsr::ii<DataType, 3>& strain) noexcept {
  auto strain_trace_term = get<0, 0>(strain);
  for (size_t d = 1; d < 3; ++d) {
    strain_trace_term += strain.get(d, d);
  }
  strain_trace_term *= 2. / 3.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      result->get(i, j) = 2. * strain.get(i, j);
    }
    result->get(i, i) -= strain_trace_term;
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                \
  template void longitudinal_operator(                      \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,      \
      const tnsr::ii<DTYPE(data), 3>& strain,               \
      const tnsr::II<DTYPE(data), 3>& inv_metric) noexcept; \
  template void longitudinal_operator_flat_cartesian(       \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,      \
      const tnsr::ii<DTYPE(data), 3>& strain) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond

}  // namespace Xcts
