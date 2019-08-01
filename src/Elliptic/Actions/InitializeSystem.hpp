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
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

/*!
 * \brief Initializes the DataBox tags related to the system
 *
 * The system fields are initially set to zero here.
 *
 * With:
 * - `fluxes_tag` = `db::add_tag_prefix<Tags::Flux, variables_tag,
 * tmpl::size_t<volume_dim>, Frame::Inertial>`
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 *   - `variables_tag`
 *   - `compute_fluxes`
 *   - `compute_sources`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *   - All items required by the added compute tags
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *   - `db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, fields_tag>`
 *   - `db::add_tag_prefix<::Tags::FixedSource, fields_tag>`
 *   - `variables_tag`
 *   - `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 *   variables_tag>`
 *   - `fluxes_tag`
 *   - `db::add_tag_prefix<Tags::Source, variables_tag>`
 *   - `db::add_tag_prefix<Tags::div, fluxes_tag>`
 */
struct InitializeSystem {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using operator_applied_to_fields_tag =
        db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
    using fixed_sources_tag =
        db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
    using vars_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using operator_applied_to_vars_tag =
        db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, vars_tag>;
    using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                          tmpl::size_t<Dim>, Frame::Inertial>;
    using inv_jacobian_tag =
        ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                                ::Tags::Coordinates<Dim, Frame::Logical>>;

    using simple_tags =
        db::AddSimpleTags<fields_tag, operator_applied_to_fields_tag,
                          fixed_sources_tag, vars_tag,
                          operator_applied_to_vars_tag>;
    using compute_tags = db::AddComputeTags<
        // First-order fluxes and sources
        typename system::compute_fluxes, typename system::compute_sources,
        // Divergence of the system fluxes for the elliptic operator
        ::Tags::DivCompute<fluxes_tag, inv_jacobian_tag>>;

    const auto& mesh = db::get<Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Set initial data to zero. Non-zero initial data would require us to also
    // compute the linear operator applied to the the initial data.
    db::item_type<fields_tag> fields{num_grid_points, 0.};
    db::item_type<operator_applied_to_fields_tag> operator_applied_to_fields{
        num_grid_points, 0.};

    auto fixed_sources = variables_from_tagged_tuple(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords,
                       db::get_variables_tags_list<fixed_sources_tag>{}));

    // Initialize the variables for the elliptic solve. Their initial value is
    // determined by the linear solver. The value is also updated by the linear
    // solver in every step.
    db::item_type<vars_tag> vars{num_grid_points};

    // Initialize the linear operator applied to the variables. It needs no
    // initial value, but is computed in every step of the elliptic solve.
    db::item_type<operator_applied_to_vars_tag> operator_applied_to_vars{
        num_grid_points};

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeSystem, simple_tags,
                                             compute_tags>(
            std::move(box), std::move(fields),
            std::move(operator_applied_to_fields), std::move(fixed_sources),
            std::move(vars), std::move(operator_applied_to_vars)));
  }
};

}  // namespace Actions
}  // namespace elliptic
