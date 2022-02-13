// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/expressions/Matrix.h>
#include <blaze/math/expressions/Vector.h>
#include <pup.h>

namespace LinearSolver::Serial {

// See Eigen/src/IterativeLinearSolvers/IncompleteLUT.h
template <typename ValueType>
struct IncompleteLU {
  template <bool SO, typename Tag>
  void compute(const blaze::CompressedMatrix<ValueType, SO, Tag>& matrix) {
    static_assert(SO == blaze::rowMajor);
    lu_ = matrix;
    for (size_t i = 0; i < matrix.rows(); ++i) {
      for (auto it_k = matrix.begin(i); it_k != matrix.end(i); ++it_k) {
        const size_t k = it_k->index();
        if (k >= i) {
          break;
        }
        lu_(i, k) /= lu_(k, k);
        for (auto it_j = matrix.begin(i); it_j != matrix.end(i); ++it_j) {
          const size_t j = it_j->index();
          if (j < k) {
            continue;
          }
          lu_(i, j) -= lu_(i, k) * lu_(k, j);
        }
      }
    }
  }

  template <typename VT1, bool TF1, typename VT2, bool TF2>
  void solve(blaze::Vector<VT1, TF1>& solution,
             const blaze::Vector<VT2, TF2>& rhs) {
    (void)solution;
    (void)rhs;
  }

  void pup(PUP::er& /*p*/) {}

 private:
  blaze::CompressedMatrix<ValueType, blaze::rowMajor> lu_{};
};

}  // namespace LinearSolver::Serial
