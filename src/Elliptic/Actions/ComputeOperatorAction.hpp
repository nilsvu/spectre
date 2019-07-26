// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace tuples {
template <typename...>
class TaggedTuple;  // IWYU pragma: keep
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Compute the bulk contribution to the operator represented by the
 * `StepPrefixTag` applied to the `VarsTag`.
 *
 * This action computes \f$A(u)=-\partial_i F^i(u) + S(u)\f$, where \f$F^i\f$
 * and \f$S\f$ are the fluxes and sources of the system of first-order PDEs,
 * respectively. They are defined such that \f$A(u(x))=f(x)\f$ is the full
 * system of equations, with \f$f(x)\f$ representing sources that are
 * independent of the variables \f$u\f$. In a DG setting, boundary contributions
 * can be added to \f$A(u)\f$ in a subsequent action to build the full DG
 * operator action.
 *
 * We generally build the operator $A(u)$ to perform elliptic solver iterations.
 * This action can be used to build operators for both the linear solver and the
 * nonlinear solver iterations by providing the appropriate `StepPrefixTag` and
 * `VarsTag`.
 *
 * With:
 * - `operator_tag` = `db::add_tag_prefix<StepPrefixTag, VarsTag>`
 * - `fluxes_tag` = `db::add_tag_prefix<::Tags::Flux, VarsTag,
 * tmpl::size_t<Dim>, Frame::Inertial>`
 * - `div_fluxes_tag` = `db::add_tag_prefix<::Tags::div, fluxes_tag>`
 * - `sources_tag` = `db::add_tag_prefix<::Tags::Source, VarsTag>`
 *
 * Uses:
 * - DataBox:
 *   - `div_fluxes_tag`
 *   - `sources_tag`
 *
 * DataBox changes:
 * - Modifies:
 *   - `operator_tag`
 */
template <size_t Dim, template <typename> class StepPrefixTag, typename VarsTag>
struct ComputeOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using operator_tag = db::add_tag_prefix<StepPrefixTag, VarsTag>;
    using fluxes_tag = db::add_tag_prefix<::Tags::Flux, VarsTag,
                                          tmpl::size_t<Dim>, Frame::Inertial>;
    using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
    using sources_tag = db::add_tag_prefix<::Tags::Source, VarsTag>;
    db::mutate_apply<tmpl::list<operator_tag>,
                     tmpl::list<div_fluxes_tag, sources_tag>>(
        [](const gsl::not_null<db::item_type<operator_tag>*>
               operator_applied_to_vars,
           const db::item_type<div_fluxes_tag>& div_fluxes,
           const db::item_type<sources_tag>& sources) {
          *operator_applied_to_vars = -1. * div_fluxes + sources;
        },
        make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace elliptic
