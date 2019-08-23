// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/MirrorVariables.hpp"
#include "Domain/Tags.hpp"
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

namespace detail {
template <size_t Dim, typename System, bool Enabled>
struct InhomogeneousBoundaryComputeTags {
  using type = db::AddComputeTags<>;
};
template <size_t Dim, typename System>
struct InhomogeneousBoundaryComputeTags<Dim, System, true> {
  using analytic_fields_tag =
      db::add_tag_prefix<::Tags::Analytic, typename System::fields_tag>;
  using type = db::AddComputeTags<
      // We slice the analytic solutions to the interior boundary and
      // compute their normal-dot-fluxes for imposing inhomogeneous
      // Dirichlet boundary conditions.
      ::Tags::Slice<::Tags::BoundaryDirectionsExterior<Dim>,
                    analytic_fields_tag>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<Dim>,
                                   typename System::compute_analytic_fluxes>,
      ::Tags::InterfaceComputeItem<
          ::Tags::BoundaryDirectionsExterior<Dim>,
          ::Tags::NormalDotCompute<
              db::add_tag_prefix<::Tags::NormalFlux, analytic_fields_tag,
                                 tmpl::size_t<Dim>, Frame::Inertial>,
              Dim, Frame::Inertial>>>;
};
}  // namespace detail

// System is needed for boundary conditions
template <typename BoundaryScheme, typename System,
          bool PrepareInhomogenousBoundaryConditions = true>
struct InitializeFluxes {
 private:
  static constexpr size_t Dim = BoundaryScheme::volume_dim;
  using vars_tag = typename BoundaryScheme::variables_tag;
  using normal_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalFlux, vars_tag, tmpl::size_t<Dim>,
                         Frame::Inertial>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags = db::AddSimpleTags<>;
    using compute_tags = tmpl::append<
        tmpl::list<
            // We slice the fluxes and their divergences to all interior
            // faces
            ::Tags::Slice<::Tags::InternalDirections<Dim>, normal_fluxes_tag>,
            ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>,
                          normal_fluxes_tag>,
            ::Tags::Slice<::Tags::InternalDirections<Dim>, fluxes_tag>,
            ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>, fluxes_tag>,
            // ::Tags::Slice<::Tags::InternalDirections<Dim>, div_fluxes_tag>,
            // ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>,
            //               div_fluxes_tag>,
            // For the strong flux lifting scheme we need the interface
            // normal dotted into the fluxes.
            ::Tags::InterfaceComputeItem<
                ::Tags::InternalDirections<Dim>,
                ::Tags::NormalDotCompute<normal_fluxes_tag, Dim,
                                         Frame::Inertial>>,
            ::Tags::InterfaceComputeItem<
                ::Tags::BoundaryDirectionsInterior<Dim>,
                ::Tags::NormalDotCompute<normal_fluxes_tag, Dim,
                                         Frame::Inertial>>,
            ::Tags::InterfaceComputeItem<
                ::Tags::InternalDirections<Dim>,
                ::Tags::NormalDotCompute<fluxes_tag, Dim, Frame::Inertial>>,
            ::Tags::InterfaceComputeItem<
                ::Tags::BoundaryDirectionsInterior<Dim>,
                ::Tags::NormalDotCompute<fluxes_tag, Dim, Frame::Inertial>>
            // We mirror the system variables to the exterior (ghost)
            // faces to impose homogeneous (zero) boundary conditions.
            // Non-zero boundary conditions are handled as contributions
            // to the source term during initialization.
            // ::Tags::Slice<::Tags::BoundaryDirectionsInterior<Dim>, vars_tag>,
            // ::Tags::InterfaceComputeItem<
            //     ::Tags::BoundaryDirectionsExterior<Dim>,
            //     ::Tags::MirrorVariables<
            //         Dim, ::Tags::BoundaryDirectionsInterior<Dim>, vars_tag,
            //         typename System::primal_variables>>,
            // On exterior (ghost) boundary faces we compute the fluxes
            // from the data that is being mirrored there to impose
            // homogeneous Dirichlet boundary conditions. Then, we
            // compute their normal-dot-fluxes. The flux divergences are
            // sliced from the volume.
            // ::Tags::InterfaceComputeItem<
            //     ::Tags::BoundaryDirectionsExterior<Dim>,
            //     typename System::compute_fluxes>,
            // ::Tags::InterfaceComputeItem<
            //     ::Tags::BoundaryDirectionsExterior<Dim>,
            //     ::Tags::NormalDotCompute<vars_tag, Dim, Frame::Inertial>>,
            // ::Tags::Slice<::Tags::BoundaryDirectionsExterior<Dim>,
            //               div_fluxes_tag>
            >,
        tmpl::type_from<detail::InhomogeneousBoundaryComputeTags<
            Dim, System, PrepareInhomogenousBoundaryConditions>>>;

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeFluxes, simple_tags,
                                             compute_tags>(std::move(box)));
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
