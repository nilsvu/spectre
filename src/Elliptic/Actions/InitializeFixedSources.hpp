// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

/*!
 * \brief Initialize the "fixed sources" of the elliptic equations, i.e. their
 * variable-independent source term \f$f(x)\f$
 *
 * This action initializes \f$f(x)\f$ in an elliptic system of PDEs \f$-div(F) +
 * S = f(x)\f$.
 *
 * Uses:
 * - System:
 *   - `primal_fields`
 * - DataBox:
 *   - `BackgroundTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `db::wrap_tags_in<::Tags::FixedSource, primal_fields>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename System, typename BackgroundTag>
struct InitializeFixedSources {
 private:
  using fixed_sources_tag = ::Tags::Variables<
      db::wrap_tags_in<::Tags::FixedSource, typename System::primal_fields>>;

 public:
  using simple_tags = tmpl::list<fixed_sources_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& background = db::get<BackgroundTag>(box);

    // Retrieve the fixed-sources of the elliptic system from the background,
    // which (along with the boundary conditions) define the problem we want to
    // solve.
    auto fixed_sources = variables_from_tagged_tuple(background.variables(
        inertial_coords, typename fixed_sources_tag::type::tags_list{}));

    if (db::get<elliptic::dg::Tags::Massive>(box)) {
      const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
      const auto& det_inv_jacobian = db::get<
          domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(box);
      fixed_sources /= get(det_inv_jacobian);
      // This is the full mass matrix (no diagonal approximation). The lifting
      // operation uses the diagonal approximation. Problem?
      const Matrix identity{};
      auto mass_matrices = make_array<Dim>(std::cref(identity));
      for (size_t d = 0; d < Dim; ++d) {
        gsl::at(mass_matrices, d) =
            Spectral::mass_matrix(mesh.slice_through(d));
      }
      fixed_sources =
          apply_matrices(mass_matrices, fixed_sources, mesh.extents());
    }

    ::Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(fixed_sources));
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
