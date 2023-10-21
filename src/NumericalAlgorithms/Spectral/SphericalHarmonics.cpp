// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Spectral {

// Algorithms to compute spherical harmonic basis functions
// These functions specialize the templates declared in `Spectral.hpp`.

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::SphericalHarmonic, Quadrature::Gauss>(const size_t num_points) {
  return compute_collocation_points_and_weights<Basis::Legendre,
                                                Quadrature::Gauss>(num_points);
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::SphericalHarmonic, Quadrature::Equiangular>(
    const size_t num_points) {
  return compute_collocation_points_and_weights<Basis::FiniteDifference,
                                                Quadrature::CellCentered>(
      num_points);
}

template <>
DataVector compute_basis_function_value<Basis::SphericalHarmonic>(
    const size_t /*k*/, const DataVector& /*x*/) {
  ERROR("not implemented");
}

template <>
DataVector compute_inverse_weight_function_values<Basis::SphericalHarmonic>(
    const DataVector& /*x*/) {
  ERROR("not implemented");
}

template <>
double compute_basis_function_normalization_square<Basis::SphericalHarmonic>(
    const size_t /*k*/) {
  ERROR("not implemented");
}

template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

template <>
Matrix spectral_indefinite_integral_matrix<Basis::SphericalHarmonic>(
    const size_t /*num_points*/) {
  ERROR("not implemented");
}

}  // namespace Spectral
