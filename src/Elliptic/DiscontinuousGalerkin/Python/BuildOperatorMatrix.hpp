// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/CreateElements.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/DgElementArray.hpp"
#include "NumericalAlgorithms/LinearSolver/BuildMatrix.hpp"

namespace py = pybind11;

namespace elliptic::dg::py_bindings {

// For the matrix representation we need to choose a particular ordering of
// the element data

template <size_t Dim, typename TagsList>
struct OperatorIterator {
  OperatorIterator(
      std::unordered_map<ElementId<Dim>, Variables<TagsList>>& vars,
      std::vector<ElementId<Dim>> element_ids, size_t element_index = 0,
      size_t data_index = 0)
      : vars_(vars),
        element_ids_(std::move(element_ids)),
        element_index_(element_index),
        data_index_(data_index),
        index_(0) {}

  OperatorIterator& operator++() {
    ++data_index_;
    ++index_;
    if (data_index_ == vars_.at(element_ids_.at(element_index_)).size()) {
      ++element_index_;
      data_index_ = 0;
    }
    return *this;
  }
  double& operator*() {
    return vars_.at(element_ids_.at(element_index_)).data()[data_index_];
  }
  size_t index() { return index_; }
  OperatorIterator begin() { return {vars_, element_ids_}; }
  OperatorIterator end() {
    return {vars_, element_ids_, element_ids_.size(), 0};
  }

  friend bool operator==(const OperatorIterator& lhs,
                         const OperatorIterator& rhs) {
    return lhs.element_index_ == rhs.element_index_ and
           lhs.data_index_ == rhs.data_index_;
  }
  friend bool operator!=(const OperatorIterator& lhs,
                         const OperatorIterator& rhs) {
    return not(lhs == rhs);
  }

 private:
  std::unordered_map<ElementId<Dim>, Variables<TagsList>>& vars_;
  std::vector<ElementId<Dim>> element_ids_;
  size_t element_index_;
  size_t data_index_;
  size_t index_;
};

template <size_t Dim, typename TagsList>
struct OrderedElementData : std::map<ElementId<Dim>, Variables<TagsList>> {
  // TODO: Add iterators over all data
};

template <typename System, bool Linearized,
          typename Initializers = tmpl::list<>, typename... GlobalCacheTags,
          size_t Dim = System::volume_dim>
Matrix build_operator_matrix(const DomainCreator<Dim>& domain_creator,
                             const double penalty_parameter, const bool massive,
                             const tuples::TaggedTuple<GlobalCacheTags...>&
                                 global_cache = tuples::TaggedTuple<>{}) {
  static_assert(Linearized,
                "Can only build a matrix representation for linear operators. "
                "Set 'Linearized' to 'True'.");
  const auto dg_element_array =
      create_elements<System, Linearized, Initializers>(domain_creator,
                                                        global_cache);
  using OperandBuffer = OrderedElementData<Dim, typename System::primal_fields>;
  using ResultBuffer =
      OrderedElementData<Dim, db::wrap_tags_in<DgOperatorAppliedTo,
                                               typename System::primal_fields>>;
  OperandBuffer operand_buffer{};
  ResultBuffer result_buffer{};
  std::unordered_map<ElementId<Dim>, Workspace<System>> workspace{};
  // Count total operator size and initialize buffers
  size_t operator_size = 0;
  for (const auto& [element_id, dg_element] : dg_element_array) {
    const size_t num_points =
        get<domain::Tags::Mesh<Dim>>(dg_element).number_of_grid_points();
    operand_buffer.emplace(element_id,
                           typename OperandBuffer::mapped_type{num_points, 0.});
    result_buffer.emplace(element_id,
                          typename ResultBuffer::mapped_type{num_points, 0.});
    operator_size += operand_buffer.at(element_id).size();
  }
  // Build operator matrix by applying the operator to unit vectors
  Matrix operator_matrix{operator_size, operator_size, 0.};
  LinearSolver::Serial::build_matrix(
      make_not_null(&operator_matrix), make_not_null(&operand_buffer),
      make_not_null(&result_buffer),
      [&dg_element_array, &workspace, &penalty_parameter, &massive](
          const gsl::not_null<ResultBuffer*> local_result_buffer,
          const OperandBuffer& local_operand_buffer) {
        elliptic::dg::py_bindings::apply_operator<System, Linearized>(
            local_result_buffer, make_not_null(&workspace), dg_element_array,
            local_operand_buffer, penalty_parameter, massive);
      });
  return operator_matrix;
}

}  // namespace elliptic::dg::py_bindings
