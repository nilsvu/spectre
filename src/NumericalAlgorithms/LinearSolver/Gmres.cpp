// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"

#include <tuple>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"

namespace LinearSolver {
namespace gmres_detail {

std::pair<DenseVector<double>, double> minimal_residual(
    const DenseMatrix<double>& orthogonalization_history,
    const double initial_residual_magnitude) noexcept {
  // Perform a QR decomposition of the Hessenberg matrix that was built during
  // the orthogonalization
  DenseMatrix<double> qr_Q;
  DenseMatrix<double> qr_R;
  blaze::qr(orthogonalization_history, qr_Q, qr_R);
  // Compute the residual vector from the QR decomposition
  DenseVector<double> beta(orthogonalization_history.rows(), 0.);
  beta[0] = initial_residual_magnitude;
  const DenseVector<double> minres =
      blaze::inv(qr_R) * blaze::trans(qr_Q) * beta;
  return {minres, blaze::length(beta - orthogonalization_history * minres)};
}

}
}  // namespace LinearSolver
