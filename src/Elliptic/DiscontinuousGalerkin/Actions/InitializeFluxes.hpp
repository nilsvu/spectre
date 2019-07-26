// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Conservative/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic {
namespace dg {
namespace Actions {

struct InitializeFluxes {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t Dim = system::volume_dim;
    using vars_tag = typename system::variables_tag;
    using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                          tmpl::size_t<Dim>, Frame::Inertial>;
    using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
    using analytic_fields_tag =
        db::add_tag_prefix<::Tags::Analytic, typename system::fields_tag>;
    using exterior_vars_tag =
        ::Tags::Interface<::Tags::BoundaryDirectionsExterior<Dim>, vars_tag>;

    using simple_tags = db::AddSimpleTags<exterior_vars_tag>;
    using compute_tags = db::AddComputeTags<
        // We slice the fluxes and their divergences to all interior faces
        ::Tags::Slice<::Tags::InternalDirections<Dim>, fluxes_tag>,
        ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>, fluxes_tag>,
        ::Tags::Slice<::Tags::InternalDirections<Dim>, div_fluxes_tag>,
        ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>, div_fluxes_tag>,
        // For the strong flux lifting scheme we need the interface normal
        // dotted into the fluxes.
        ::Tags::InterfaceComputeItem<
            ::Tags::InternalDirections<Dim>,
            ::Tags::ComputeNormalDotFlux<vars_tag, Dim, Frame::Inertial>>,
        ::Tags::InterfaceComputeItem<
            ::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::ComputeNormalDotFlux<vars_tag, Dim, Frame::Inertial>>,
        // We slice the analytic solutions to the interior boundary and compute
        // their normal-dot-fluxes for imposing inhomogeneous Dirichlet boundary
        // conditions.
        ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>,
                      analytic_fields_tag>,
        ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsInterior<Dim>,
                                     typename system::compute_analytic_fluxes>,
        ::Tags::InterfaceComputeItem<
            ::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::ComputeNormalDotFlux<analytic_fields_tag, Dim,
                                         Frame::Inertial>>,
        // On exterior (ghost) boundary faces we compute the fluxes from the
        // data that is being set there manually to impose homogeneous Dirichlet
        // boundary conditions. Then, we compute their normal-dot-fluxes. The
        // flux divergences are sliced from the volume. We also need the system
        // variables to mirror them to the exterior.
        ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<Dim>,
                                     typename system::compute_fluxes>,
        ::Tags::InterfaceComputeItem<
            ::Tags::BoundaryDirectionsExterior<Dim>,
            ::Tags::ComputeNormalDotFlux<vars_tag, Dim, Frame::Inertial>>,
        ::Tags::Slice<::Tags::BoundaryDirectionsExterior<Dim>, div_fluxes_tag>,
        ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>, vars_tag>>;

    // Initialize the variables on the exterior (ghost) boundary faces.
    // These are stored in a simple tag and updated manually to impose boundary
    // conditions.
    db::item_type<exterior_vars_tag> exterior_boundary_vars{};
    const auto& mesh = db::get<::Tags::Mesh<Dim>>(box);
    for (const auto& direction :
         db::get<::Tags::BoundaryDirectionsExterior<Dim>>(box)) {
      exterior_boundary_vars[direction] = db::item_type<vars_tag>{
          mesh.slice_away(direction.dimension()).number_of_grid_points()};
    }

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeFluxes, simple_tags,
                                             compute_tags>(
            std::move(box), std::move(exterior_boundary_vars)));
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
