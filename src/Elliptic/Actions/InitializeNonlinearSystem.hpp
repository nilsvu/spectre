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
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

template <typename Metavariables>
struct InitializeNonlinearSystem {
 private:
  using initial_guess_tag = typename Metavariables::initial_guess_tag;

 public:
  using const_global_cache_tags = tmpl::list<initial_guess_tag>;

  template <typename DataBox, typename... InboxTags,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using nonlinear_fields_tag = typename system::fields_tag;
    using initial_nonlinear_fields_tag =
        db::add_tag_prefix<::Tags::Initial, nonlinear_fields_tag>;
    using operator_applied_to_nonlinear_fields_tag =
        db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo,
                           nonlinear_fields_tag>;
    using nonlinear_fixed_sources_tag =
        db::add_tag_prefix<::Tags::FixedSource, nonlinear_fields_tag>;
    using correction_tag = db::add_tag_prefix<NonlinearSolver::Tags::Correction,
                                              nonlinear_fields_tag>;
    using linear_operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, correction_tag>;
    using operator_applied_to_linear_operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                           linear_operand_tag>;

    using linear_fluxes_tag =
        db::add_tag_prefix<::Tags::Flux, linear_operand_tag, tmpl::size_t<Dim>,
                           Frame::Inertial>;
    using nonlinear_fluxes_tag =
        db::add_tag_prefix<::Tags::Flux, nonlinear_fields_tag,
                           tmpl::size_t<Dim>, Frame::Inertial>;
    using inv_jacobian_tag =
        ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                                ::Tags::Coordinates<Dim, Frame::Logical>>;

    const auto& mesh = db::get<::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Retrieve initial guess for the fields
    auto nonlinear_fields = variables_from_tagged_tuple(
        Parallel::get<initial_guess_tag>(cache).variables(
            inertial_coords,
            db::get_variables_tags_list<initial_nonlinear_fields_tag>{}));

    // Retrieve the sources of the elliptic system from the analytic solution,
    // which defines the problem we want to solve.
    // We need only retrieve sources for the primal fields, since the auxiliary
    // fields will never be sourced.
    db::item_type<nonlinear_fixed_sources_tag> nonlinear_fixed_sources{
        num_grid_points, 0.};
    nonlinear_fixed_sources.assign_subset(
        Parallel::get<initial_guess_tag>(cache).variables(
            inertial_coords,
            db::wrap_tags_in<::Tags::FixedSource,
                             typename system::primal_fields>{}));

    // The nonlinear solver computes this in each step
    db::item_type<operator_applied_to_nonlinear_fields_tag>
        operator_applied_to_nonlinear_fields{num_grid_points};

    // The nonlinear solver initializes this (to zero)
    db::item_type<correction_tag> correction{num_grid_points};

    // Initialize the variables for the elliptic solve. Their initial value is
    // determined by the linear solver. The value is also updated by the linear
    // solver in every step.
    db::item_type<linear_operand_tag> linear_operand{num_grid_points};

    // Initialize the linear operator applied to the variables. It needs no
    // initial value, but is computed in every step of the elliptic solve.
    db::item_type<operator_applied_to_linear_operand_tag>
        operator_applied_to_linear_operand{num_grid_points};

    using nonlinear_fluxes_compute_tag =
        elliptic::Tags::FirstOrderFluxesCompute<Dim, system>;
    using nonlinear_sources_compute_tag =
        elliptic::Tags::FirstOrderSourcesCompute<system>;
    using linear_fluxes_compute_tag = elliptic::Tags::FirstOrderFluxesCompute<
        Dim, typename system::linearized_system>;
    using linear_sources_compute_tag = elliptic::Tags::FirstOrderSourcesCompute<
        typename system::linearized_system>;

    using compute_tags = db::AddComputeTags<
        typename Metavariables::initial_guess::compute_tags,
        nonlinear_fluxes_compute_tag, nonlinear_sources_compute_tag,
        linear_fluxes_compute_tag, linear_sources_compute_tag,
        ::Tags::DivCompute<linear_fluxes_tag, inv_jacobian_tag>,
        ::Tags::DivCompute<nonlinear_fluxes_tag, inv_jacobian_tag>>;

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeNonlinearSystem,
            db::AddSimpleTags<nonlinear_fields_tag, nonlinear_fixed_sources_tag,
                              operator_applied_to_nonlinear_fields_tag,
                              correction_tag, linear_operand_tag,
                              operator_applied_to_linear_operand_tag>,
            compute_tags>(std::move(box),
                          db::item_type<nonlinear_fields_tag>(nonlinear_fields),
                          std::move(nonlinear_fixed_sources),
                          std::move(operator_applied_to_nonlinear_fields),
                          std::move(correction), std::move(linear_operand),
                          std::move(operator_applied_to_linear_operand)));
  }
};

}  // namespace Actions
}  // namespace elliptic
