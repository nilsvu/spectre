// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Initializes the DataBox tags related to the system
 *
 * The system fields are initially set to zero here.
 *
 * Uses:
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 *   - `variables_tag`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *   - `variables_tag`
 *   - `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 *   variables_tag>`
 */
template <typename Metavariables, typename = cpp17::void_t<>>
struct System {
 private:
  using system = typename Metavariables::system;
  using fields_tag = typename system::fields_tag;
  using fields_operator_tag =
      db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using variables_tag = typename system::variables_tag;
  using variables_operator_tag =
      db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo,
                         variables_tag>;

 public:
  static constexpr size_t Dim = system::volume_dim;
  using simple_tags = db::AddSimpleTags<fields_tag, fields_operator_tag,
                                        variables_tag, variables_operator_tag>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
    const size_t num_grid_points =
        db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();

    // Set initial data to zero. Non-zero initial data would require the
    // linear solver initialization to also compute the Ax term.
    db::item_type<fields_tag> fields{num_grid_points, 0.};
    db::item_type<fields_operator_tag> operator_applied_to_fields{
        num_grid_points, 0.};

    // Initialize the variables for the elliptic solve. Their initial value is
    // determined by the linear solver. The value is also updated by the linear
    // solver in every step.
    db::item_type<variables_tag> vars{num_grid_points};

    // Initialize the linear operator applied to the variables. It needs no
    // initial value, but is computed in every step of the elliptic solve.
    db::item_type<variables_operator_tag> operator_applied_to_vars{
        num_grid_points};

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(fields),
        std::move(operator_applied_to_fields), std::move(vars),
        std::move(operator_applied_to_vars));
  }
};

template <typename Metavariables>
struct System<Metavariables,
              cpp17::void_t<typename Metavariables::nonlinear_solver>> {
 private:
  using system = typename Metavariables::system;
  using fields_tag = typename system::nonlinear_fields_tag;
  using initial_fields_tag = db::add_tag_prefix<::Tags::Initial, fields_tag>;
  using fields_operator_tag =
      db::add_tag_prefix<::NonlinearSolver::Tags::OperatorAppliedTo,
                         fields_tag>;
  using correction_tag = typename system::fields_tag;
  using operand_tag = typename system::variables_tag;
  using operand_operator_tag =
      db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, operand_tag>;

 public:
  static constexpr size_t Dim = system::volume_dim;
  using simple_tags =
      db::AddSimpleTags<fields_tag, fields_operator_tag, correction_tag,
                        operand_tag, operand_operator_tag>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    const size_t num_grid_points =
        db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();
    const auto& inertial_coords =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Retrieve initial guess for the fields from the cache
    db::item_type<initial_fields_tag> fields{num_grid_points};
    fields.assign_subset(
        Parallel::get<typename Metavariables::initial_guess_tag>(cache)
            .variables(inertial_coords,
                       db::get_variables_tags_list<initial_fields_tag>{}));

    // The nonlinear solver computes this in each step
    db::item_type<fields_operator_tag> fields_operator{num_grid_points};

    // The nonlinear solver initializes this (to zero)
    db::item_type<correction_tag> correction{num_grid_points};

    // Initialize the variables for the elliptic solve. Their initial value is
    // determined by the linear solver. The value is also updated by the linear
    // solver in every step.
    db::item_type<operand_tag> operand{num_grid_points};

    // Initialize the linear operator applied to the variables. It needs no
    // initial value, but is computed in every step of the elliptic solve.
    db::item_type<operand_operator_tag> operand_operator{num_grid_points};

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), db::item_type<fields_tag>{fields},
        std::move(fields_operator), std::move(correction), std::move(operand),
        std::move(operand_operator));
  }
};

}  // namespace Initialization
}  // namespace Elliptic
