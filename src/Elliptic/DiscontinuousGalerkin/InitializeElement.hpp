// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Domain.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/Initialization/BoundaryConditions.hpp"
#include "Elliptic/Initialization/Derivatives.hpp"
#include "Elliptic/Initialization/DiscontinuousGalerkin.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "Elliptic/Initialization/FluxLifting.hpp"
#include "Elliptic/Initialization/Interface.hpp"
#include "Elliptic/Initialization/Source.hpp"
#include "Elliptic/Initialization/System.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeFluxes.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace tuples {
template <typename... Tags>
class TaggedTuple;  // IWYU pragma: keep
}  // namespace tuples
/// \endcond

namespace Elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Initializes the DataBox of each element in the DgElementArray
 *
 * The following initializers are chained together (in this order):
 *
 * - `Elliptic::Initialization::Domain`
 * - `Elliptic::Initialization::System`
 * - `Elliptic::Initialization::Source`
 * - `Elliptic::Initialization::Derivatives`
 * - `Elliptic::Initialization::Interface`
 * - `Elliptic::Initialization::BoundaryConditions`
 * - `linear_solver::tags` (depends on the boundary-modified source)
 * - `Elliptic::Initialization::DiscontinuousGalerkin`
 */
template <size_t Dim>
struct InitializeElement {
  template <class Metavariables>
  using return_tag_list = tmpl::append<
      // Simple tags
      typename Elliptic::Initialization::Domain<Dim>::simple_tags,
      typename Elliptic::Initialization::System<Metavariables>::simple_tags,
      db::AddSimpleTags<NonlinearSolver::Tags::IterationId,
                        LinearSolver::Tags::IterationId>,
      typename Elliptic::Initialization::Source<Metavariables>::simple_tags,
      typename Elliptic::Initialization::Derivatives<
          typename Metavariables::system>::simple_tags,
      typename Elliptic::Initialization::Interface<
          typename Metavariables::system>::simple_tags,
      typename Elliptic::Initialization::DiscontinuousGalerkin<
          Metavariables>::simple_tags,
      typename Elliptic::Initialization::FluxLifting<
          typename Metavariables::nonlinear_flux_lifting_scheme>::simple_tags,
      typename Elliptic::Initialization::FluxLifting<
          typename Metavariables::linear_flux_lifting_scheme>::simple_tags,
      typename Metavariables::nonlinear_solver::tags::simple_tags,
      typename Metavariables::linear_solver::tags::simple_tags,
      // Compute tags
      typename Elliptic::Initialization::Domain<Dim>::compute_tags,
      typename Elliptic::Initialization::System<Metavariables>::compute_tags,
      db::AddComputeTags<
          NonlinearSolver::Tags::NextIterationIdCompute,
          LinearSolver::Tags::NextIterationIdCompute,
          Elliptic::Tags::IterationIdCompute<NonlinearSolver::Tags::IterationId,
                                             LinearSolver::Tags::IterationId>>,
      typename Elliptic::Initialization::Source<Metavariables>::compute_tags,
      typename Elliptic::Initialization::Derivatives<
          typename Metavariables::system>::compute_tags,
      typename Elliptic::Initialization::Interface<
          typename Metavariables::system>::compute_tags,
      typename Elliptic::Initialization::DiscontinuousGalerkin<
          Metavariables>::compute_tags,
      typename Elliptic::Initialization::FluxLifting<
          typename Metavariables::nonlinear_flux_lifting_scheme>::compute_tags,
      typename Elliptic::Initialization::FluxLifting<
          typename Metavariables::linear_flux_lifting_scheme>::compute_tags,
      typename Metavariables::nonlinear_solver::tags::compute_tags,
      typename Metavariables::linear_solver::tags::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList action_list_meta,
                    const ParallelComponent* const parallel_component_meta,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain) noexcept {
    using system = typename Metavariables::system;
    auto domain_box = Elliptic::Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto system_box =
        Elliptic::Initialization::System<Metavariables>::initialize(
            std::move(domain_box), cache);
    auto temporal_id_box = db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<NonlinearSolver::Tags::IterationId,
                          LinearSolver::Tags::IterationId>,
        db::AddComputeTags<NonlinearSolver::Tags::NextIterationIdCompute,
                           LinearSolver::Tags::NextIterationIdCompute,
                           Elliptic::Tags::IterationIdCompute<
                               NonlinearSolver::Tags::IterationId,
                               LinearSolver::Tags::IterationId>>>(
        std::move(system_box), 0_st, 0_st);
    auto source_box =
        Elliptic::Initialization::Source<Metavariables>::initialize(
            std::move(temporal_id_box), cache);
    auto deriv_box = Elliptic::Initialization::Derivatives<
        typename Metavariables::system>::initialize(std::move(source_box));
    auto face_box = Elliptic::Initialization::Interface<system>::initialize(
        std::move(deriv_box));
    auto dg_box = Elliptic::Initialization::DiscontinuousGalerkin<
        Metavariables>::initialize(std::move(face_box), initial_extents);
    auto nonlinear_flux_lifting_box = Elliptic::Initialization::FluxLifting<
        typename Metavariables::nonlinear_flux_lifting_scheme>::
        initialize(std::move(dg_box), initial_extents);
    // Compute initial nonlinear operator
    db::mutate_apply<
        typename system::compute_nonlinear_operator_action::return_tags,
        typename system::compute_nonlinear_operator_action::argument_tags>(
        typename system::compute_nonlinear_operator_action{},
        make_not_null(&nonlinear_flux_lifting_box));
    // Impose homogeneous b.c. on nonlinear operator and contribute inhom. b.c.
    // to source
    auto tmp = get<0>(
        ::dg::Actions::ComputeFluxes<
            ::Tags::BoundaryDirectionsInterior<Dim>,
            typename system::nonlinear_normal_dot_fluxes>::
            apply(nonlinear_flux_lifting_box, inboxes, cache, array_index,
                  action_list_meta, parallel_component_meta));
    auto tmp2 = get<0>(
        Elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
            typename Metavariables::nonlinear_flux_lifting_scheme,
            typename system::impose_boundary_conditions_on_fields>::
            apply(tmp, inboxes, cache, array_index, action_list_meta,
                  parallel_component_meta));
    db::mutate_apply_cache<
        typename Metavariables::nonlinear_flux_lifting_scheme>(
        make_not_null(&tmp2), cache, true);
    db::mutate_apply_cache<Elliptic::Initialization::BoundaryConditions<
        Metavariables, typename Metavariables::nonlinear_flux_lifting_scheme,
        typename Metavariables::system::impose_boundary_conditions_on_fields>>(
        make_not_null(&tmp2), cache);
    auto linear_flux_lifting_box = Elliptic::Initialization::FluxLifting<
        typename Metavariables::linear_flux_lifting_scheme>::
        initialize(std::move(tmp2), initial_extents);
    auto nonlinear_solver_box =
        Metavariables::nonlinear_solver::tags::initialize(
            std::move(linear_flux_lifting_box), cache, array_index,
            parallel_component_meta);
    auto linear_solver_box = Metavariables::linear_solver::tags::initialize(
        std::move(nonlinear_solver_box), cache, array_index,
        parallel_component_meta);
    return std::make_tuple(std::move(linear_solver_box));
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace Elliptic
