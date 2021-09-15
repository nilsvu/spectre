// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <Eigen/SparseCore>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "DataStructures/DenseMatrix.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace LinearSolver::Serial {

namespace detail {
CREATE_IS_CALLABLE(reset)
CREATE_IS_CALLABLE_V(reset)
}  // namespace detail

/*!
 * \brief Construct explicit matrix representation by "sniffing out" the
 * operator, i.e. feeding it unit vectors
 *
 * Assumes the `matrix`, the `operand_buffer` and the `result_buffer` are sized
 * correctly on input, and the `operand_buffer` is zero.
 */
template <typename LinearOperator, typename OperandType, typename ResultType,
          typename... OperatorArgs>
void build_matrix(const gsl::not_null<Eigen::SparseMatrix<double>*> matrix,
                  const gsl::not_null<OperandType*> operand_buffer,
                  const gsl::not_null<ResultType*> result_buffer,
                  const LinearOperator& linear_operator,
                  const std::tuple<OperatorArgs...>& operator_args) {
  size_t i = 0;
  // Re-using the iterators for all operator invocations
  auto result_iterator_begin = result_buffer->begin();
  auto result_iterator_end = result_buffer->end();
  std::vector<Eigen::Triplet<double>> sparse_entries{};
  sparse_entries.reserve(square(static_cast<size_t>(matrix->rows())));
  for (double& unit_vector_data : *operand_buffer) {
    // Add a 1 at the unit vector location i
    unit_vector_data = 1.;
    // Invoke the operator on the unit vector
    std::apply(
        linear_operator,
        std::tuple_cat(std::forward_as_tuple(result_buffer, *operand_buffer),
                       operator_args));
    // Set the unit vector back to zero
    unit_vector_data = 0.;
    // Reset the iterator by calling its `reset` member function or by
    // re-creating it
    if constexpr (detail::is_reset_callable_v<
                      decltype(result_iterator_begin)>) {
      result_iterator_begin.reset();
    } else {
      result_iterator_begin = result_buffer->begin();
      result_iterator_end = result_buffer->end();
    }
    // Store the result in column i of the matrix
    for (size_t j = 0; j < static_cast<size_t>(matrix->rows()); ++j) {
      if (not equal_within_roundoff(*result_iterator_begin, 0.)) {
        sparse_entries.emplace_back(j, i, *result_iterator_begin);
      }
      ++result_iterator_begin;
    }
    ++i;
  }
  matrix->setFromTriplets(sparse_entries.begin(), sparse_entries.end());
}

}  // namespace LinearSolver::Serial
