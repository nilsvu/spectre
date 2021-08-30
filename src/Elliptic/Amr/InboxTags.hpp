// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <tuple>

#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::amr::detail::InboxTags {

struct ErrorMeasurement : Parallel::InboxInserters::Value<ErrorMeasurement> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, std::tuple<double, Convergence::HasConverged>>;
};

}  // namespace elliptic::amr::detail::InboxTags
