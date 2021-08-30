// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/Amr/ElementActions.hpp"
#include "Elliptic/Amr/ErrorMonitor.hpp"
#include "Elliptic/Amr/Observe.hpp"
#include "IO/Observer/Helpers.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::amr {

template <typename Metavariables, size_t Dim, typename ArraySectionIdTag = void>
struct Amr {
  using component_list = tmpl::list<detail::ErrorMonitor<Metavariables, Dim>>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<observe_detail::reduction_data>>;

  template <typename InitializationActions, typename SolveActions>
  using iterate =
      tmpl::list<detail::Prepare<Dim, ArraySectionIdTag>, SolveActions,
                 detail::ContributeError<Dim, ArraySectionIdTag>,
                 detail::RefineMesh<Dim, ArraySectionIdTag>,
                 InitializationActions,
                 detail::Complete<Dim, ArraySectionIdTag>>;
};

}  // namespace elliptic::amr
