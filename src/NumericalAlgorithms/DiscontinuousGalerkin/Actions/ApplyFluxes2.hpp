// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"  // IWYU pragma: keep // for db::item_type<Tags::Mortars<...>>
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace Tags {
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace dg {
namespace Actions {

template <typename FluxLiftingScheme>
struct ApplyFluxes {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate_apply<typename FluxLiftingScheme::return_tags,
                     typename FluxLiftingScheme::argument_tags>(
        FluxLiftingScheme{}, make_not_null(&box),
        get<typename Metavariables::normal_dot_numerical_flux>(cache));
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace dg
