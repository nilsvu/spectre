// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"

#include <array>
#include <cmath>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace PowerMonitors {

template <typename VectorType, size_t Dim>
void power_monitors(const gsl::not_null<std::array<VectorType, Dim>*> result,
                    const VectorType& u, const Mesh<Dim>& mesh) {
  // Get modal coefficients
  const auto modal_coefficients = to_modal_coefficients(u, mesh);
  using ValueType = typename VectorType::value_type;

  ValueType slice_sum = 0.0;
  size_t n_slice = 0;
  size_t n_stripe = 0;
  for (size_t sliced_dim = 0; sliced_dim < Dim; ++sliced_dim) {
    n_slice = mesh.extents().slice_away(sliced_dim).product();
    n_stripe = mesh.extents(sliced_dim);

    gsl::at(*result, sliced_dim).destructive_resize(n_stripe);

    for (size_t index = 0; index < n_stripe; ++index) {
      slice_sum = 0.0;
      for (SliceIterator si(mesh.extents(), sliced_dim, index); si; ++si) {
        slice_sum += square(modal_coefficients[si.volume_offset()]);
      }
      slice_sum /= n_slice;
      slice_sum = sqrt(slice_sum);

      gsl::at(*result, sliced_dim)[index] = slice_sum;
    }
  }
}

template <typename VectorType, size_t Dim>
std::array<VectorType, Dim> power_monitors(const VectorType& u,
                                           const Mesh<Dim>& mesh) {
  std::array<VectorType, Dim> result{};
  power_monitors(make_not_null(&result), u, mesh);
  return result;
}

// The power_monitor argument should be made of type ModalVector
// when pybindings for ModalVector are enabled
template <typename VectorType>
double relative_truncation_error(const VectorType& power_monitor,
                                 const size_t num_modes_to_use) {
  ASSERT(
      num_modes_to_use <= power_monitor.size(),
      "Number of modes needs less or equal than the number of power monitors");
  ASSERT(2_st <= num_modes_to_use,
         "Number of modes needs to be larger or equal than 2.");
  const size_t last_index = num_modes_to_use - 1;
  const double max_mode = max(abs(power_monitor));
  const double cutoff =
      100. * std::numeric_limits<double>::epsilon() * max_mode;
  // If the last two or more modes are zero, assume that the function is
  // represented exactly and return a relative truncation error of zero.
  // Just one zero mode is not enough to make this assumption, as the function
  // could have zero modes by symmetry.
  if (num_modes_to_use >= 2 and abs(power_monitor[last_index]) < cutoff and
      abs(power_monitor[last_index - 1]) < cutoff) {
    return -log10(cutoff) + 2.;
  }
  // Compute weighted average and total sum in the current dimension
  using ValueType = typename VectorType::value_type;
  ValueType weighted_average = 0.0;
  double weight_sum = 0.0;
  double weight_value = 0.0;
  for (size_t index = 0; index <= last_index; ++index) {
    const auto mode = power_monitor[index];
    if (abs(mode) < cutoff) {
      // Ignore modes below this cutoff, so modes that are zero (e.g. by
      // symmetry) don't make us underestimate the truncation error.
      continue;
    }
    // Compute current weight
    weight_value = exp(-square(static_cast<double>(last_index - index) - 0.5));
    // Add weighted power monitor
    weighted_average += weight_value * log10(mode);
    // Add term to weighted sum
    weight_sum += weight_value;
  }
  weighted_average /= weight_sum;

  // Maximum between the first two power monitors
  double leading_term = std::max(abs(power_monitor[0]), abs(power_monitor[1]));
  ASSERT(not(leading_term == 0.0),
         "The leading power monitor term is zero bitwise.");
  leading_term = log10(leading_term);

  // Compute relative truncation error
  // TODO complex?
  return abs(leading_term - weighted_average);
}

template <typename VectorType, size_t Dim>
std::array<double, Dim> relative_truncation_error(
    const VectorType& tensor_component, const Mesh<Dim>& mesh) {
  std::array<double, Dim> result{};
  const auto modes = power_monitors(tensor_component, mesh);
  for (size_t d = 0; d < Dim; ++d) {
    const auto& modes_d = gsl::at(modes, d);
    gsl::at(result, d) =
        pow(10.0, -relative_truncation_error(modes_d, modes_d.size()));
  }
  return result;
}

template <typename VectorType, size_t Dim>
std::array<double, Dim> absolute_truncation_error(
    const VectorType& tensor_component, const Mesh<Dim>& mesh) {
  std::array<double, Dim> result{};
  const auto modes = power_monitors(tensor_component, mesh);
  // Use infinity norm to estimate the order of magnitude of the variable
  const double umax = max(abs(tensor_component));
  double relative_truncation_error_in_d = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    const auto& modes_d = gsl::at(modes, d);
    // Compute relative truncation error
    relative_truncation_error_in_d =
        relative_truncation_error(modes_d, modes_d.size());
    // Compute absolute truncation error estimate
    gsl::at(result, d) =
        umax * pow(10.0, -1.0 * relative_truncation_error_in_d);
  }
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template std::array<DTYPE(data), DIM(data)> power_monitors(             \
      const DTYPE(data) & u, const Mesh<DIM(data)>& mesh);                \
  template void power_monitors(                                           \
      const gsl::not_null<std::array<DTYPE(data), DIM(data)>*> result,    \
      const DTYPE(data) & u, const Mesh<DIM(data)>& mesh);                \
  template std::array<double, DIM(data)> relative_truncation_error(       \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh); \
  template std::array<double, DIM(data)> absolute_truncation_error(       \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace PowerMonitors
